#!/usr/bin/env python3
# =====================================================================
#  fpga_npu_gui.py  -  PC <-> FPGA 실시간 UART 데모 GUI
#
#  동작:
#    1) GTSRB 테스트 이미지 폴더 열기 -> 썸네일 그리드 표시
#       (같은 폴더에 GT-*.csv 있으면 ROI 좌표 자동 사용)
#    2) 썸네일 클릭 -> 학습과 동일하게 전처리(28x28 흑백) -> Q7 양자화
#    3) UART로 784바이트 전송 -> FPGA 추론
#    4) 42바이트 결과패킷 수신 [0xAA, class, logit0..9(int32 LE)]
#    5) 예측 클래스명 + confidence(softmax) 막대 표시
#
#  필요 패키지:  pip install pyqt5 pyserial opencv-python numpy
# =====================================================================
import sys
import glob
import os
import time

import numpy as np
import cv2
import serial
import serial.tools.list_ports

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "gtsrb_10class_grayscale_c5_c6_d16_three_seed_results",
    "grayscale_c5_c6_d16_split100_train100",
    "best_model_grayscale_c5_c6_d16_split100_train100.keras"
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QComboBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog, QScrollArea,
    QFrame, QSizePolicy, QProgressBar
)

# ---------------------------------------------------------------------
#  설정 (학습 코드와 반드시 동일하게 유지)
# ---------------------------------------------------------------------
IMAGE_SIZE = 28
MIN_ROI_SIZE = 8
ROI_MARGIN_RATIO = 0.0

# 모델 출력 인덱스 0~9 -> 표지판 이름 (학습 SELECTED_CLASSES 순서)
CLASS_NAMES = [
    "양보", "정지", "진입 금지", "일반 위험 도로", "도로 공사 중",
    "어린이 횡단 주의", "우회전 지시", "좌회전 지시", "직진 지시", "회전교차로",]

# GTSRB ClassId -> 한국어 이름
GTSRB_ID_TO_NAME = {
    13: "양보", 14: "정지", 17: "진입 금지", 18: "일반 위험 도로",
    25: "도로 공사 중", 28: "어린이 횡단 주의", 33: "우회전 지시",
    34: "좌회전 지시", 35: "직진 지시", 40: "회전교차로",
}

BAUD = 115200
IMG_BYTES = 784                 # 28*28
PACKET_BYTES = 42               # 0xAA + class + 10*int32


# ---------------------------------------------------------------------
#  전처리 (train_gtsrb_rgb_c5_c6_d16.py 의 resize_with_padding 과 동일)
# ---------------------------------------------------------------------
def resize_with_padding(image, target_size=28):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    xs = (target_size - nw) // 2
    ys = (target_size - nh) // 2
    canvas[ys:ys + nh, xs:xs + nw] = resized
    return canvas


def preprocess(path, roi=None):
    """이미지 -> 28x28 흑백 uint8 (0~255). roi=(x1,y1,x2,y2) 또는 None(전체)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)        # BGR
    if img is None:
        return None
    H, W = img.shape[:2]
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(int(x1), W - 1)); y1 = max(0, min(int(y1), H - 1))
        x2 = max(0, min(int(x2), W - 1)); y2 = max(0, min(int(y2), H - 1))
        if x2 < x1 or y2 < y1:
            return None
        img = img[y1:y2 + 1, x1:x2 + 1]
    if img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 흑백
    return resize_with_padding(gray, IMAGE_SIZE)    # 28x28 uint8


def quantize_q7(gray28):
    """28x28 uint8(0~255) -> Q7 784바이트(0~127). FPGA가 받는 입력."""
    # 학습: pixel/255.0 정규화 -> 우리: *128 해서 Q7
    q = np.round(gray28.astype(np.float32) / 255.0 * 128.0)
    q = np.clip(q, 0, 127).astype(np.uint8)
    return q.flatten().tobytes()                    # 행우선(HWC) 784바이트


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ---------------------------------------------------------------------
#  시리얼 워커 스레드 (UART 송수신은 블로킹이므로 별도 스레드)
# ---------------------------------------------------------------------
class SerialWorker(QThread):
    finished_ok = pyqtSignal(int, object, float)  # (pred_class, logits, elapsed_ms)
    failed = pyqtSignal(str)

    def __init__(self, port, payload):
        super().__init__()
        self.port = port
        self.payload = payload               # 784바이트

    def run(self):
        try:
            with serial.Serial(self.port, BAUD, timeout=3) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                t0 = time.perf_counter()
                ser.write(self.payload)      # 784바이트 전송
                ser.flush()

                # 42바이트 결과패킷 수신
                pkt = ser.read(PACKET_BYTES)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if len(pkt) != PACKET_BYTES:
                    self.failed.emit(
                        f"패킷 수신 실패: {len(pkt)}/{PACKET_BYTES} 바이트")
                    return
                if pkt[0] != 0xAA:
                    self.failed.emit(f"헤더 오류: 0x{pkt[0]:02X} (기대 0xAA)")
                    return
                pred = pkt[1]
                logits = np.frombuffer(pkt[2:42], dtype="<i4").astype(np.int64)
                self.finished_ok.emit(pred, logits, elapsed_ms)
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------------------------------------------------
#  클릭 가능한 썸네일 라벨
# ---------------------------------------------------------------------
class Thumb(QLabel):
    def __init__(self, path, roi, on_click):
        super().__init__()
        self.path = path
        self.roi = roi
        self.on_click = on_click
        pix = QPixmap(path)
        if pix.isNull():
            # ppm 등 QPixmap이 못 읽으면 cv2로 읽어 변환
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is not None:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                h, w, _ = im.shape
                pix = QPixmap.fromImage(
                    QImage(im.data, w, h, 3 * w, QImage.Format_RGB888))
        self.setPixmap(pix.scaled(64, 64, Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation))
        self.setFixedSize(72, 72)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("border:1px solid #aaa; margin:2px;")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        self.on_click(self.path, self.roi)


# ---------------------------------------------------------------------
#  메인 윈도우
# ---------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPGA NPU 교통표지판 분류 데모")
        self.resize(950, 750)
        self.worker = None
        try:
            self.keras_model = tf.keras.models.load_model(MODEL_PATH)
            self.keras_model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
        except Exception as e:
            self.keras_model = None
            print(f"모델 로드 실패: {e}")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ---- 상단: 포트 선택 + 폴더 열기 ----
        top = QHBoxLayout()
        self.port_box = QComboBox()
        self.refresh_ports()
        btn_refresh = QPushButton("포트 새로고침")
        btn_refresh.clicked.connect(self.refresh_ports)
        btn_open = QPushButton("이미지 폴더 열기")
        btn_open.clicked.connect(self.open_folder)
        top.addWidget(QLabel("COM 포트:"))
        top.addWidget(self.port_box, 1)
        top.addWidget(btn_refresh)
        top.addWidget(btn_open)
        root.addLayout(top)

        # ---- 중앙: 썸네일 그리드(좌) + 결과패널(우) ----
        mid = QHBoxLayout()

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        mid.addWidget(scroll, 3)

        # 결과 패널
        right = QVBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(240, 240)
        self.preview.setStyleSheet("border:1px solid #888; background:#000;")
        self.preview.setAlignment(Qt.AlignCenter)
        right.addWidget(self.preview)

        lbl_truth_header = QLabel("<b>실제 정답</b>")
        lbl_truth_header.setStyleSheet("font-size:20px;")
        right.addWidget(lbl_truth_header)
        self.truth_lbl = QLabel("—")
        self.truth_lbl.setStyleSheet("font-size:26px; font-weight:bold; color:#1a7a1a;")
        right.addWidget(self.truth_lbl)

        lbl_pred_header = QLabel("<b>FPGA 예측 결과</b>")
        lbl_pred_header.setStyleSheet("font-size:20px;")
        right.addWidget(lbl_pred_header)
        self.result_lbl = QLabel("—")
        self.result_lbl.setStyleSheet("font-size:26px; font-weight:bold;")
        right.addWidget(self.result_lbl)

        self.fpga_time_lbl = QLabel("")
        self.fpga_time_lbl.setStyleSheet("font-size:13px; color:#555;")
        right.addWidget(self.fpga_time_lbl)
        right.addSpacing(10)

        lbl_py_header = QLabel("<b>Python 예측 결과</b>")
        lbl_py_header.setStyleSheet("font-size:20px;")
        right.addWidget(lbl_py_header)
        self.py_result_lbl = QLabel("—")
        self.py_result_lbl.setStyleSheet("font-size:26px; font-weight:bold; color:#1a3a8a;")
        right.addWidget(self.py_result_lbl)
        self.py_time_lbl = QLabel("")
        self.py_time_lbl.setStyleSheet("font-size:13px; color:#555;")
        right.addWidget(self.py_time_lbl)
        right.addSpacing(16)

        # confidence 상위 막대들
        self.bars = []
        for i in range(3):
            row = QHBoxLayout()
            rank = QLabel(f"{i+1}.")
            rank.setFixedWidth(22)
            rank.setStyleSheet("font-size:15px; font-weight:bold;")
            name = QLabel("—"); name.setFixedWidth(130)
            name.setStyleSheet("font-size:15px;")
            bar = QProgressBar(); bar.setMaximum(100); bar.setTextVisible(True)
            bar.setFixedHeight(28)
            bar.setStyleSheet("font-size:14px;")
            row.addWidget(rank); row.addWidget(name); row.addWidget(bar)
            right.addLayout(row)
            self.bars.append((name, bar))

        self.status = QLabel("폴더를 열고 이미지를 클릭하세요.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#555;")
        right.addWidget(self.status)
        right.addStretch(1)

        rw = QWidget(); rw.setLayout(right)
        rw.setFixedWidth(420)
        mid.addWidget(rw)
        root.addLayout(mid, 1)

    # ---- 포트 목록 ----
    def refresh_ports(self):
        self.port_box.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_box.addItem(p.device)
        if not ports:
            self.port_box.addItem("(포트 없음)")

    # ---- 폴더 열기 + 썸네일 채우기 ----
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "GTSRB 이미지 폴더 선택")
        if not folder:
            return
        # 이미지 목록
        exts = ("*.ppm", "*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(folder, e))
        files.sort()
        if not files:
            self.status.setText("이미지가 없습니다 (.ppm/.png/.jpg).")
            return

        # GTSRB ROI csv(GT-*.csv) 있으면 읽기 {filename: (x1,y1,x2,y2)}
        roi_map = {}
        self.label_map = {}
        # 현재 폴더 + 상위 폴더 + 상위상위 폴더에서 GT-*.csv 탐색
        search_dirs = [folder,
                       os.path.dirname(folder),
                       os.path.dirname(os.path.dirname(folder))]
        csv_paths = []
        for d in search_dirs:
            csv_paths += glob.glob(os.path.join(d, "GT-*.csv"))
        for csv_path in csv_paths:
            try:
                import csv as _csv
                with open(csv_path, newline="") as f:
                    rd = _csv.DictReader(f, delimiter=";")
                    cols = {c.lower(): c for c in rd.fieldnames}
                    for r in rd:
                        fn = r[cols["filename"]]
                        roi_map[fn] = (
                            int(r[cols["roi.x1"]]), int(r[cols["roi.y1"]]),
                            int(r[cols["roi.x2"]]), int(r[cols["roi.y2"]]))
                        if "classid" in cols:
                            cid = int(r[cols["classid"]])
                            self.label_map[fn] = GTSRB_ID_TO_NAME.get(cid, f"기타({cid})")
            except Exception:
                pass

        # 그리드 비우기
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()

        # 썸네일 채우기
        cols = 6
        for i, path in enumerate(files[:500]):
            fn = os.path.basename(path)
            roi = roi_map.get(fn, None)
            thumb = Thumb(path, roi, self.on_thumb_click)
            self.grid.addWidget(thumb, i // cols, i % cols)
        self.status.setText(
            f"{min(len(files),120)}장 로드됨 "
            f"(ROI csv: {'있음' if roi_map else '없음'}). 클릭해서 추론.")

    # ---- 썸네일 클릭 -> 전처리 -> 전송 ----
    def on_thumb_click(self, path, roi):
        gray = preprocess(path, roi)
        if gray is None:
            self.status.setText("전처리 실패(ROI/파일 확인).")
            return

        # 28x28 미리보기 (5배 확대)
        big = cv2.resize(gray, (140, 140), interpolation=cv2.INTER_NEAREST)
        qimg = QImage(big.data, 140, 140, 140, QImage.Format_Grayscale8)
        self.preview.setPixmap(QPixmap.fromImage(qimg))

        # 실제 정답 표시
        fn = os.path.basename(path)
        truth = getattr(self, "label_map", {}).get(fn, "정보 없음")
        self.truth_lbl.setText(f"➜ {truth}")

        # Python 추론
        if self.keras_model is not None:
            inp = gray.astype(np.float32) / 255.0
            inp = inp.reshape(1, 28, 28, 1)
            t0 = time.perf_counter()
            py_pred = np.argmax(self.keras_model.predict(inp, verbose=0))
            py_ms = (time.perf_counter() - t0) * 1000
            py_name = CLASS_NAMES[py_pred] if 0 <= py_pred < 10 else f"?{py_pred}"
            self.py_result_lbl.setText(f"➜ {py_name}(#{py_pred})")
            self.py_time_lbl.setText(f"추론 시간: {py_ms:.1f} ms")
        else:
            self.py_result_lbl.setText("모델 없음")
            self.py_time_lbl.setText("")

        port = self.port_box.currentText()
        if port.startswith("("):
            self.status.setText("COM 포트를 선택하세요.")
            return

        payload = quantize_q7(gray)      # 784바이트
        self.status.setText("전송 중... FPGA 추론 대기")
        self.result_lbl.setText("...")

        # 워커 스레드로 송수신
        self.worker = SerialWorker(port, payload)
        self.worker.finished_ok.connect(self.on_result)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    # ---- 결과 수신 ----
    def on_result(self, pred, logits, fpga_ms):
        # Q14 logit -> 실수 -> softmax
        real = logits.astype(np.float64) / 16384.0
        prob = softmax(real)
        order = np.argsort(prob)[::-1]

        name = CLASS_NAMES[pred] if 0 <= pred < 10 else f"?{pred}"
        self.result_lbl.setText(f"➜ {name}(#{pred})")
        self.fpga_time_lbl.setText(f"추론 시간: {fpga_ms:.1f} ms")

        for k, (lbl, bar) in enumerate(self.bars):
            idx = int(order[k])
            lbl.setText(CLASS_NAMES[idx])
            bar.setValue(int(round(prob[idx] * 100)))
            bar.setFormat(f"{prob[idx]*100:.1f}%")
        self.status.setText("")

    def on_fail(self, msg):
        self.result_lbl.setText("실패")
        self.status.setText(f"오류: {msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
