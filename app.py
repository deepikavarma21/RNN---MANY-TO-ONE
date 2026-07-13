"""
Streamlit app: Student Performance Predictor (High / Medium / Low)
Loads the trained RNN (LSTM) model + preprocessing objects produced by train_rnn.py

Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# ---------------------------------------------------------------------------
# Load model + preprocessors (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("student_performance_rnn.keras")
    with open("preprocessors.pkl", "rb") as f:
        prep = pickle.load(f)
    return model, prep

model, prep = load_artifacts()
seq_scaler = prep["seq_scaler"]
ohe = prep["ohe"]
label_encoder = prep["label_encoder"]
categorical_cols = prep["categorical_cols"]
categorical_options = prep["categorical_options"]

GRADE_EMOJI = {"High": "🏆", "Medium": "📘", "Low": "⚠️"}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write("Predicts whether a student will fall into the **High**, **Medium**, or **Low** "
         "performance category, using an RNN (LSTM) trained on math/reading/writing scores "
         "and demographic context.")

st.subheader("Scores")
col1, col2, col3 = st.columns(3)
with col1:
    math_score = st.slider("Math score", 0, 100, 70)
with col2:
    reading_score = st.slider("Reading score", 0, 100, 70)
with col3:
    writing_score = st.slider("Writing score", 0, 100, 70)

st.subheader("Student background")
inputs = {}
c1, c2 = st.columns(2)
cols_ui = [c1, c2, c1, c2, c1]
for col_name, ui_col in zip(categorical_cols, cols_ui):
    label = col_name.replace("_", " ").title()
    inputs[col_name] = ui_col.selectbox(label, categorical_options[col_name])

st.divider()

if st.button("🚀 Predict Performance", type="primary", use_container_width=True):
    # Build sequence input: [math, reading, writing] scaled, shape (1, 3, 1)
    seq_raw = np.array([[math_score, reading_score, writing_score]], dtype="float32")
    seq_scaled = seq_scaler.transform(seq_raw).reshape(1, 3, 1)

    # Build static (demographic) input using the same OneHotEncoder as training
    static_raw = [[inputs[col] for col in categorical_cols]]
    static_encoded = ohe.transform(static_raw)

    # Predict
    probs = model.predict([seq_scaled, static_encoded], verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    st.markdown(f"## {GRADE_EMOJI[pred_label]} Predicted category: **{pred_label}**")
    st.progress(confidence / 100)
    st.write(f"Confidence: **{confidence:.1f}%**")

    st.subheader("Class probabilities")
    for cls, p in zip(label_encoder.classes_, probs):
        st.write(f"{GRADE_EMOJI[cls]} {cls}: {p*100:.1f}%")

st.divider()
st.caption(
    "Note: this model was trained on a dataset where the target label (High/Medium/Low) "
    "was derived directly from these same three scores, so it mainly demonstrates the "
    "modeling pipeline rather than genuine independent prediction. For real forecasting, "
    "train on independent predictors (attendance, study hours, past-term grades, etc.) "
    "against outcomes from a *different* term."
)
