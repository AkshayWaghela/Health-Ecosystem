import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD FILES
# ==============================
df_user = pd.read_csv("df_user.csv")
df_user = df_user.loc[:, ~df_user.columns.str.contains('^Unnamed')]

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Smart Health Risk Predictor")

# ==============================
# USER INPUTS (SIMPLE)
# ==============================
age = st.number_input("Age", 10, 100, 25)

gender = st.selectbox("Gender", ["Male", "Female"])
sex = 1 if gender == "Male" else 0

bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

smoking = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

hr = st.number_input("Avg Heart Rate", 40, 180, 75)
steps = st.number_input("Daily Steps", 0, 20000, 8000)
sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
water = st.number_input("Water Intake (L)", 0.0, 5.0, 2.0)

# ==============================
# SIMILAR USER FUNCTION
# ==============================
def get_similar_users(df, age, sex, bmi, smoking):

    df_filtered = df.copy()

    df_filtered = df_filtered[
        (df_filtered['age_first'] >= age - 5) &
        (df_filtered['age_first'] <= age + 5)
    ]

    df_filtered = df_filtered[df_filtered['sex_first'] == sex]

    df_filtered = df_filtered[
        (df_filtered['bmi_mean'] >= bmi - 2) &
        (df_filtered['bmi_mean'] <= bmi + 2)
    ]

    df_filtered = df_filtered[
        df_filtered['smoking_status_first'] == smoking
    ]

    return df_filtered

# ==============================
# ANALYZE
# ==============================
if st.button("Analyze"):

    # Get similar users
    similar_users = get_similar_users(df_user, age, sex, bmi, smoking)

    if len(similar_users) < 10:
        similar_users = df_user  # fallback

    avg_vals = similar_users.mean(numeric_only=True)

    # ==============================
    # CREATE INPUT
    # ==============================
    new_user = pd.DataFrame([{
        'age_first': age,
        'sex_first': sex,
        'bmi_mean': bmi,
        'smoking_status_first': smoking,

        'family_history_cvd_first': avg_vals['family_history_cvd_first'],
        'fitness_level_first': avg_vals['fitness_level_first'],

        'avg_heart_rate_mean': hr,
        'avg_heart_rate_std': avg_vals['avg_heart_rate_std'],
        'avg_heart_rate_max': hr,

        'resting_hr_mean': avg_vals['resting_hr_mean'],
        'resting_hr_std': avg_vals['resting_hr_std'],

        'hrv_mean': avg_vals['hrv_mean'],
        'hrv_std': avg_vals['hrv_std'],

        'steps_mean': steps,
        'steps_std': avg_vals['steps_std'],
        'calories_burned_mean': avg_vals['calories_burned_mean'],

        'sleep_hours_mean': sleep,
        'sleep_hours_std': avg_vals['sleep_hours_std'],
        'sleep_efficiency_mean': avg_vals['sleep_efficiency_mean'],

        'spo2_mean': avg_vals['spo2_mean'],
        'body_temp_c_mean': avg_vals['body_temp_c_mean'],
        'fatigue_score_mean': avg_vals['fatigue_score_mean'],

        'water_intake_l_mean': water
    }])

    # ==============================
    # FEATURES (STRICT)
    # ==============================
    features = [
        'age_first','sex_first','bmi_mean','smoking_status_first',
        'family_history_cvd_first','fitness_level_first',
        'avg_heart_rate_mean','avg_heart_rate_std','avg_heart_rate_max',
        'resting_hr_mean','resting_hr_std',
        'hrv_mean','hrv_std',
        'steps_mean','steps_std','calories_burned_mean',
        'sleep_hours_mean','sleep_hours_std','sleep_efficiency_mean',
        'spo2_mean','body_temp_c_mean','fatigue_score_mean',
        'water_intake_l_mean'
    ]

    # ==============================
    # MODEL PREDICTION
    # ==============================
    X_input = new_user[features]
    X_scaled = scaler.transform(X_input)

    prob = model.predict_proba(X_scaled)[0][1]
    score = 100 - prob * 100

    # ==============================
    # CATEGORY
    # ==============================
    if prob < 0.3:
        category = "Healthy"
    elif prob < 0.7:
        category = "Moderate Risk"
    else:
        category = "High Risk"

    # ==============================
    # RANKINGS (ORIGINAL DATA)
    # ==============================
    def percentile(col, value, reverse=False):
        p = (df_user[col] < value).mean() * 100
        return 100 - p if reverse else p

    sleep_rank = percentile('sleep_hours_mean', sleep)
    steps_rank = percentile('steps_mean', steps)
    water_rank = percentile('water_intake_l_mean', water)
    hr_rank = percentile('avg_heart_rate_mean', hr, reverse=True)

    # ==============================
    # INSIGHTS
    # ==============================
    insights = []
    suggestions = []

    if hr > 100:
        insights.append("High heart rate")
        suggestions.append("Do cardio & manage stress")

    if sleep < 6:
        insights.append("Poor sleep")
        suggestions.append("Aim for 7–8 hours sleep")

    if steps < 5000:
        insights.append("Low activity")
        suggestions.append("Walk at least 8,000 steps")

    if water < 1.5:
        insights.append("Low hydration")
        suggestions.append("Drink 2–3L water daily")

    if bmi > 25:
        insights.append("High BMI")
        suggestions.append("Improve diet & exercise")

    if len(insights) == 0:
        insights.append("Healthy pattern")
        suggestions.append("Keep maintaining lifestyle")

    # ==============================
    # DISPLAY
    # ==============================
    st.subheader("📊 Health Report")

    st.metric("Health Score", round(score, 2))
    st.metric("Risk Category", category)
    st.metric("Risk Probability", round(prob, 2))

    st.subheader("🏆 Rankings")
    st.write(f"Sleep: {round(sleep_rank,1)} percentile")
    st.write(f"Steps: {round(steps_rank,1)} percentile")
    st.write(f"Water: {round(water_rank,1)} percentile")
    st.write(f"Heart Rate: {round(hr_rank,1)} percentile")

    st.subheader("📢 Insights")
    for i in insights:
        st.write("-", i)

    st.subheader("💡 Suggestions")
    for s in suggestions:
        st.write("-", s)
