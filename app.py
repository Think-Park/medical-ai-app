import numpy as np
import joblib
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="의료 AI 데모", layout="wide")


@st.cache_resource
def load_models():
    diabetes_model = keras.models.load_model("diabetes_model.keras")
    scaler = joblib.load("scaler.pkl")
    xray_model = keras.models.load_model("xray_model.keras")
    return diabetes_model, scaler, xray_model


diabetes_model, scaler, xray_model = load_models()


def predict(*values):
    x = np.array(values, dtype=float).reshape(1, -1)
    x = scaler.transform(x)
    return float(diabetes_model.predict(x, verbose=0)[0][0])


def preprocess(img):
    img = img.convert("RGB").resize((96, 96))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)          # (1, 96, 96, 3)


def predict_xray(img):
    p = float(xray_model.predict(preprocess(img), verbose=0)[0][0])
    return {"폐렴": p, "정상": 1 - p}


# 위험도 단계: (단계 이름, 색 상자 함수)
def risk_level(p):
    if p < 0.3:
        return "낮음", st.success
    elif p < 0.7:
        return "보통", st.warning
    else:
        return "높음", st.error


# 흑백 사진인지 검사: R·G·B 채널 차이가 거의 없으면 흑백
def is_grayscale(img, threshold=8):
    if img.mode in ("L", "LA", "1", "I", "I;16", "F"):
        return True
    x = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.int16)
    diff = (np.abs(x[..., 0] - x[..., 1]) + np.abs(x[..., 1] - x[..., 2])).mean()
    return diff < threshold     # JPG 압축으로 생기는 작은 차이는 허용


# 폐렴 판독 단계
def xray_level(p):
    if p >= 0.65:
        return "폐렴 의심", st.error
    elif p <= 0.35:
        return "정상 가능성 높음", st.success
    else:
        return "판독 보류: 재검사 권장", st.warning


# ── 사이드바: 검사 수치 8개 (피마 데이터 범위) ──
st.sidebar.header("검사 수치 입력")
preg = st.sidebar.number_input("임신 횟수",          0,     17,   2,     step=1)
gluc = st.sidebar.number_input("혈당 (mg/dL)",       0,     199,  148,   step=1)
bp   = st.sidebar.number_input("이완기 혈압 (mmHg)", 0,     122,  72,    step=1)
skin = st.sidebar.number_input("피부 두께 (mm)",     0,     99,   35,    step=1)
ins  = st.sidebar.number_input("인슐린 (μU/mL)",     0,     846,  0,     step=1)
bmi  = st.sidebar.number_input("체질량지수 (BMI)",   0.0,   67.1, 33.6,  step=0.1,   format="%.1f")
dpf  = st.sidebar.number_input("당뇨 가족력 지수",   0.078, 2.42, 0.627, step=0.001, format="%.3f")
age  = st.sidebar.number_input("나이",               21,    81,   50,    step=1)

st.title("🩺 의료 AI 데모")
# 안전장치 ②: 항상 표시
st.caption("⚠️ 수업용 시제품입니다. 결과는 진단이 아닌 참고용입니다.")

tab1, tab2 = st.tabs(["당뇨 위험도", "X-ray 판독"])

# ── 탭 1: 당뇨 위험도 ──
with tab1:
    st.write("왼쪽 사이드바에 검사 수치를 입력하고 버튼을 누르세요.")
    if st.button("위험도 계산"):
        prob = predict(preg, gluc, bp, skin, ins, bmi, dpf, age)
        level, box = risk_level(prob)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("당뇨 위험도", f"{prob * 100:.1f}%")
        with col2:
            box(f"위험 단계: **{level}**")
        st.progress(prob, text=f"{prob * 100:.1f}%  (30% 미만 낮음 · 30~70% 보통 · 70% 이상 높음)")

        # 안전장치 ①: 애매한 구간
        if 0.4 <= prob <= 0.6:
            st.info("판단이 애매한 구간입니다. 재검사를 권장합니다.")

# ── 탭 2: X-ray 판독 ──
with tab2:
    uploaded = st.file_uploader("흉부 X-ray 사진을 올리세요", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)

        # 안전장치 ③: 컬러 사진 차단
        if not is_grayscale(img):
            st.error("흉부 X-ray(흑백) 사진만 올려 주세요")
            st.stop()

        result = predict_xray(img)
        p = result["폐렴"]
        label, box = xray_level(p)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="올린 사진", use_container_width=True)
        with col2:
            st.metric("폐렴 확률", f"{p * 100:.1f}%")
            box(f"판독: **{label}**")
            st.progress(p, text=f"폐렴 {p * 100:.1f}% · 정상 {result['정상'] * 100:.1f}%")
