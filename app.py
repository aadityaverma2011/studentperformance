import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")

st.markdown("""
    <h2 style='text-align: center; color: #4CAF50;'>🎓 Student Performance Predictor</h2>
    <p style='text-align: center;'>Predict whether a student is likely to <strong>pass</strong> or <strong>fail</strong> based on background inputs.</p>
    <hr>
""", unsafe_allow_html=True)

# Form layout
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

    with col2:
        parent_edu = st.selectbox("Parental Education Level", [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ])
        reading_score = st.slider("Reading Score", 0, 100, 50)
        writing_score = st.slider("Writing Score", 0, 100, 50)

    math_score = st.slider("Math Score", 0, 100, 50)

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = {
            "gender": gender,
            "lunch": lunch,
            "test preparation course": test_prep,
            "parental level of education": parent_edu,
            "math score": math_score,
            "reading score": reading_score,
            "writing score": writing_score
        }

        # Encode categorical inputs
        for key in ["gender", "lunch", "test preparation course", "parental level of education"]:
            le = label_encoders[key]
            input_data[key] = le.transform([input_data[key]])[0]

        # Prepare dataframe with correct column order
        X = pd.DataFrame([[
            input_data["gender"],
            input_data["lunch"],
            input_data["test preparation course"],
            input_data["parental level of education"],
            input_data["math score"],
            input_data["reading score"],
            input_data["writing score"]
        ]], columns=[
            "gender", "lunch", "test preparation course",
            "parental level of education", "math score",
            "reading score", "writing score"
        ])

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][prediction] * 100

        # Display Result
        if prediction == 1:
            st.success(f"🎉 The student is likely to **PASS** with {probability:.2f}% confidence.")
        else:
            st.error(f"⚠️ The student is at risk of **FAILING**. Confidence: {probability:.2f}%.")

        st.markdown("---")
