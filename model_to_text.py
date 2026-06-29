# =====================================================================
#  model_to_text.py
#  학습된 Keras .keras 모델 -> Verilog $readmemh용 .mem 파일 변환
#
#  양자화 스킴 (고정소수점)
#  ---------------------------------------------------------------
#    가중치 weight : Q7   ->  w_int = round(w * 128),  [-128,127] saturate, int8
#    바이어스 bias  : Q14  ->  b_int = round(b * 128 * 128),            int32
#    입력 픽셀      : Q7   ->  x_int = round(pixel/255 * 128), [0,127], uint8
#
#  레이어 연산 (Verilog가 그대로 재현):
#    acc(Q14) = Σ x(Q7) * w(Q7)  +  bias(Q14)
#    ReLU     = max(acc, 0)
#    재양자화  = acc >> 7         (Q14 -> Q7)   ← "÷128", 다음 레이어 입력
#    마지막층 = ReLU/시프트 없이 Q14 logit 그대로 argmax
# =====================================================================
import tensorflow as tf, numpy as np, glob, os

OUT = "mem"
os.makedirs(OUT, exist_ok=True)
SCALE = 128                      # 2^7  (Q7)

mp = glob.glob("../proj/project_8_*/**/*.keras", recursive=True)[0]
print("모델:", mp)
m = tf.keras.models.load_model(mp, compile=False)

# ---- 가중치 양자화 ----
def q_w(w): return np.clip(np.round(w * SCALE),       -128, 127).astype(np.int64)  # Q7  int8
def q_b(b): return np.round(b * SCALE * SCALE).astype(np.int64)                    # Q14 int32

def hex8(v):   return format(int(v) & 0xFF,        '02x')   # int8  two's complement
def hex32(v):  return format(int(v) & 0xFFFFFFFF,  '08x')   # int32 two's complement

# ---- conv weight: (kh,kw,ic,oc) -> 주소순서 oc,kh,kw,ic ----
def write_conv(name, W, B):
    kh,kw,Cin,Cout = W.shape
    Wq, Bq = q_w(W), q_b(B)
    with open(f"{OUT}/{name}_w.mem","w") as f:
        for oc in range(Cout):
            for a in range(kh):
                for b in range(kw):
                    for ic in range(Cin):
                        f.write(hex8(Wq[a,b,ic,oc])+"\n")
    with open(f"{OUT}/{name}_b.mem","w") as f:
        for oc in range(Cout): f.write(hex32(Bq[oc])+"\n")
    over = int(np.sum((Wq>127)|(Wq<-128)))   # saturate 후엔 0
    print(f"  {name}: W {Wq.size}개(={Cout}*{kh}*{kw}*{Cin}), B {Cout}개  "
          f"[clip된 가중치 {int(np.sum((np.round(W*SCALE)>127)|(np.round(W*SCALE)<-128)))}개]")

# ---- dense weight: (in,out) -> 주소순서 j(out),i(in) ----
def write_dense(name, W, B):
    N,M = W.shape
    Wq, Bq = q_w(W), q_b(B)
    with open(f"{OUT}/{name}_w.mem","w") as f:
        for j in range(M):
            for i in range(N):
                f.write(hex8(Wq[i,j])+"\n")
    with open(f"{OUT}/{name}_b.mem","w") as f:
        for j in range(M): f.write(hex32(Bq[j])+"\n")
    print(f"  {name}: W {Wq.size}개(={M}*{N}), B {M}개  "
          f"[clip된 가중치 {int(np.sum((np.round(W*SCALE)>127)|(np.round(W*SCALE)<-128)))}개]")

print("\n.mem 파일 생성:")
write_conv ("conv1", *m.get_layer('conv2d').get_weights())
write_conv ("conv2", *m.get_layer('conv2d_1').get_weights())
write_dense("dense1",*m.get_layer('dense').get_weights())
write_dense("dense2",*m.get_layer('dense_1').get_weights())

# ===== 골든 테스트벡터 생성 (Verilog 검증용) =====
# 정수 골든 모델
c1w,c1b=m.get_layer('conv2d').get_weights();   C1W,C1B=q_w(c1w),q_b(c1b)
c2w,c2b=m.get_layer('conv2d_1').get_weights(); C2W,C2B=q_w(c2w),q_b(c2b)
d1w,d1b=m.get_layer('dense').get_weights();     D1W,D1B=q_w(d1w),q_b(d1b)
d2w,d2b=m.get_layer('dense_1').get_weights();   D2W,D2B=q_w(d2w),q_b(d2b)
def conv_same(x,W,B):
    H,Wd,Cin=x.shape; Co=W.shape[3]; out=np.zeros((H,Wd,Co),np.int64)
    xp=np.zeros((H+2,Wd+2,Cin),np.int64); xp[1:-1,1:-1,:]=x
    for oh in range(H):
        for ow in range(Wd):
            for oc in range(Co):
                acc=0
                for a in range(3):
                    for b in range(3):
                        for ic in range(Cin):
                            acc+=xp[oh+a,ow+b,ic]*W[a,b,ic,oc]
                acc+=B[oc]; acc=max(acc,0); out[oh,ow,oc]=acc>>7
    return out
def mp2(x):
    H,Wd,C=x.shape; o=np.zeros((H//2,Wd//2,C),np.int64)
    for i in range(H//2):
        for j in range(Wd//2):
            for c in range(C):
                o[i,j,c]=max(x[2*i,2*j,c],x[2*i+1,2*j,c],x[2*i,2*j+1,c],x[2*i+1,2*j+1,c])
    return o
def dense(x,W,B,relu):
    M=W.shape[1]; o=np.zeros(M,np.int64)
    for j in range(M):
        acc=int(B[j])
        for i in range(x.shape[0]): acc+=int(x[i])*int(W[i,j])
        o[j]=max(acc,0)>>7 if relu else acc
    return o

np.random.seed(7)
pixel=np.random.randint(0,256,(28,28,1))
x_q7=np.clip(np.round(pixel/255.0*128),0,127).astype(np.int64)
a=mp2(conv_same(x_q7,C1W,C1B)); a=mp2(conv_same(a,C2W,C2B))
a=dense(a.reshape(-1),D1W,D1B,True); logit=dense(a,D2W,D2B,False)
pred=int(np.argmax(logit))

with open(f"{OUT}/test_img.mem","w") as f:
    for v in x_q7.reshape(-1): f.write(format(int(v)&0xFF,'02x')+"\n")
with open(f"{OUT}/golden_logits.txt","w") as f:
    for v in logit: f.write(f"{int(v)}\n")
print(f"\n골든 테스트벡터: test_img.mem(784) + golden_logits.txt")
print(f"  기대 예측 클래스 = {pred},  logits(Q14) = {list(map(int,logit))}")
