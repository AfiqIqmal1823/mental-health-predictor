
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mental_health_model.h5")

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Mental Health Prediction from Sleep Patterns")

# --- Session State Init ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- Helper Function: Tips Based on Input ---
def generate_input_tips(gender, age, sleep_duration, quality_of_sleep, physical_activity, stress_level, bmi_category, daily_steps):
    tips = []

    if sleep_duration < 6:
        tips.append("😴 Try to increase your sleep to 7–9 hours consistently.")
    elif sleep_duration > 9:
        tips.append("😴 Oversleeping can also affect mood. Aim for 7–9 hours.")

    if quality_of_sleep < 4:
        tips.append("🛌 Improve your sleep quality by maintaining a regular bedtime and avoiding screens before bed.")

    if physical_activity == 0:
        tips.append("🚶 Start light activities like walking to improve mood and sleep.")

    if stress_level > 6:
        tips.append("🧘 High stress detected. Try mindfulness, breathing exercises, or short breaks during the day.")

    if daily_steps < 3000:
        tips.append("👟 Increase your daily steps — aim for at least 5000–8000 per day.")

    if bmi_category == "Obese":
        tips.append("🍎 Consider a balanced diet and moderate exercise for better weight management.")

    if age > 60 and physical_activity < 3:
        tips.append("🧓 Gentle exercises like stretching or tai chi can improve physical and mental well-being.")

    return tips

# --- Input Form ---
st.markdown("### Please input your health and lifestyle information:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 80, 25)
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
        daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)

    with col2:
        sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.0)
        quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 5)
        physical_activity = st.slider("Physical Activity Level (0–10)", 0, 10, 5)
        stress_level = st.slider("Stress Level (0–10)", 0, 10, 5)

    submitted = st.form_submit_button("🔍 Predict")

# --- Predict and Display ---
if submitted:
    gender_val = 1 if gender == "Male" else 0
    bmi_val = ["Underweight", "Normal", "Overweight", "Obese"].index(bmi_category)

    input_data = np.array([[gender_val, age, sleep_duration, quality_of_sleep,
                            physical_activity, stress_level, bmi_val, daily_steps]])
    input_data = input_data.reshape((1, 1, 8))

    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    confidence_scores = prediction.flatten()

    label_map = {0: "No Disorder", 1: "Anxiety", 2: "Depression"}
    result = label_map[predicted_class]

    # --- Rule-Based Override Logic (Hybrid Layer) ---
    override = None

    if stress_level >= 9 and quality_of_sleep <= 3:
        override = "Anxiety"
    elif sleep_duration <= 4 and physical_activity <= 2 and stress_level >= 6:
        override = "Depression"
    elif stress_level <= 3 and sleep_duration >= 7 and quality_of_sleep >= 7 and physical_activity >= 4:
        override = "No Disorder"

    if override:
        result = override  # override final label only

        # --- Blend confidence scores to reflect override ---
        idx = list(label_map.values()).index(result)
        confidence_scores = [score * 0.3 for score in confidence_scores]  # soften all scores
        confidence_scores[idx] = 0.7  # boost the overridden label

    # Save to session log
    st.session_state.logs.append({
        "Predicted": result,
        "Confidence": float(np.max(confidence_scores)),
        "Input": {
            "Gender": gender,
            "Age": age,
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": quality_of_sleep,
            "Physical Activity": physical_activity,
            "Stress Level": stress_level,
            "BMI": bmi_category,
            "Daily Steps": daily_steps
        }
    })

    # --- Model vs Final Result ---
    # --- Result Display ---
    st.subheader("🧾 Prediction Result")
    color = "🟢" if result == "No Disorder" else ("🟠" if result == "Anxiety" else "🔴")
    st.success(f"{color} **Predicted Condition:** {result}")

    # --- Suggested Actions ---
    st.markdown("### 🧭 Suggested Early Action Steps")
    if result == "No Disorder":
        st.success("✅ You're doing well! Maintain your routines and check in on your mental health regularly.")
    elif result == "Anxiety":
        st.warning("Try mindfulness, reduce screen time, maintain consistent sleep routine, and talk to someone you trust.")
    elif result == "Depression":
        st.error("Reach out to a mental health counselor, engage in light activity, avoid isolation, and track your sleep patterns.")

    # --- Tips ---
    st.markdown("### 🧪 Personalized Wellness Tips Based on Your Input")
    input_tips = generate_input_tips(gender, age, sleep_duration, quality_of_sleep,
                                     physical_activity, stress_level, bmi_category, daily_steps)
    if input_tips:
        for tip in input_tips:
            st.markdown(f"- {tip}")
    else:
        st.markdown("✅ Your lifestyle inputs look great! Keep it up.")

    # --- Charts ---
    st.markdown("### 🥧 Prediction Distribution (Pie Chart)")
    fig1, ax1 = plt.subplots()
    def autopct_format(pct): return f"{pct:.1f}%" if pct > 0 else ""
    ax1.pie(confidence_scores, labels=label_map.values(), autopct=autopct_format,
            colors=["green", "orange", "red"], startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("### 📊 Prediction Confidence (Bar Chart)")
    fig2, ax2 = plt.subplots()
    ax2.bar(label_map.values(), confidence_scores, color=["green", "orange", "red"])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Confidence")
    st.pyplot(fig2)

    # --- Input Recap ---
    st.markdown("### 📋 Your Input Summary")
    if st.session_state.logs:
        st.dataframe(st.session_state.logs[-1]["Input"], use_container_width=True)
    else:
        st.info("No input summary available yet.")

# --- Session Prediction Logs ---
if st.session_state.logs:
    st.markdown("### 📝 Session Prediction Log")
    for i, log in enumerate(st.session_state.logs[::-1]):
        st.markdown(f"**Prediction #{len(st.session_state.logs) - i}**")
        st.markdown(f"- 🧠 **Result**: `{log['Predicted']}`")
        st.markdown(f"- 📈 **Top Confidence**: `{log['Confidence']:.2f}`")
        with st.expander("📋 View Inputs"):
            st.json(log["Input"])

# --- Educational Section ---
st.markdown("---")
st.markdown("## 📚 Science-Backed Insights on Sleep & Mental Health")
st.info("""
- Poor sleep can trigger or worsen mental health conditions like anxiety, depression, and ADHD.
- Insomnia is often an early warning sign of depression and can double your risk of developing it.
- There's a two-way relationship: mental health problems can disrupt sleep, and sleep issues can worsen mental health.
- Treatments like Cognitive Behavioral Therapy for Insomnia (CBT-I), light therapy, and mindfulness are proven to help.
- Regular sleep schedules, morning light exposure, and screen-free wind-down routines improve emotional resilience.
""")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Muhammad Afiq Iqmal — CSP650 Final Year Project")
