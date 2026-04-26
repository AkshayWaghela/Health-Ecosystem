import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Health AI", layout="wide")

# ==============================
# STYLE (HEALTH THEME)
# ==============================
st.markdown("""
<style>
.main {
    background-color: #f4f8f6;
}
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
df_user = pd.read_csv("df_user.csv")
df_user = df_user.loc[:, ~df_user.columns.str.contains('^Unnamed')]

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# TITLE
# ==============================
st.title("🧠 Smart Health Risk Predictor")
st.write("Quick analysis using wearable-style health inputs")

# ==============================
# INPUT UI (HORIZONTAL)
# ==============================
st.subheader("🧾 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 100, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    hr = st.number_input("Heart Rate", 40, 180, 75)
    sleep = st.number_input("Sleep (hrs)", 0.0, 12.0, 7.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if gender == "Male" else 0

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    smoking = 1 if smoking == "Yes" else 0

    steps = st.number_input("Steps", 0, 20000, 8000)
    water = st.number_input("Water (L)", 0.0, 5.0, 2.0)

# ==============================
# SIMILAR USER FUNCTION
# ==============================
def get_similar_users(df, age, sex, bmi, smoking):

    df_filtered = df.copy()

    df_filtered = df_filtered[
        (df_filtered['age_first'] >= age-5) &
        (df_filtered['age_first'] <= age+5)
    ]

    df_filtered = df_filtered[df_filtered['sex_first'] == sex]

    df_filtered = df_filtered[
        (df_filtered['bmi_mean'] >= bmi-2) &
        (df_filtered['bmi_mean'] <= bmi+2)
    ]

    df_filtered = df_filtered[
        df_filtered['smoking_status_first'] == smoking
    ]

    return df_filtered

# ==============================
# ANALYSIS
# ==============================
if st.button("Analyze"):

    similar_users = get_similar_users(df_user, age, sex, bmi, smoking)

    if len(similar_users) < 10:
        similar_users = df_user

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
    # PREDICT
    # ==============================
    X_scaled = scaler.transform(new_user[features])
    prob = model.predict_proba(X_scaled)[0][1]
    score = 100 - prob*100

    if prob < 0.3:
        category = "Healthy"
    elif prob < 0.7:
        category = "Moderate Risk"
    else:
        category = "High Risk"

    # ==============================
    # KPI CARDS
    # ==============================
    st.subheader("📊 Health Summary")
    c1, c2, c3 = st.columns(3)

    c1.metric("Health Score", round(score,1))
    c2.metric("Risk Level", category)
    c3.metric("Risk %", round(prob*100,1))

    # ==============================
    # RANKINGS
    # ==============================
    def percentile(col, value, reverse=False):
        p = (df_user[col] < value).mean()*100
        return 100-p if reverse else p

    sleep_rank = percentile('sleep_hours_mean', sleep)
    steps_rank = percentile('steps_mean', steps)
    water_rank = percentile('water_intake_l_mean', water)
    hr_rank = percentile('avg_heart_rate_mean', hr, reverse=True)

    # ==============================
    # CHART 1
    # ==============================
    chart_df = pd.DataFrame({
        'Metric': ['Sleep', 'Steps', 'Water', 'Heart Rate'],
        'Percentile': [sleep_rank, steps_rank, water_rank, hr_rank]
    })

    st.subheader("📈 Performance vs Population")
    st.bar_chart(chart_df.set_index('Metric'))

    # ==============================
    # CHART 2
    # ==============================
    profile_df = pd.DataFrame({
        'Metric': ['Heart Rate','Steps','Sleep','Water'],
        'Value': [hr, steps/1000, sleep, water]
    })

    st.subheader("🧭 Your Health Profile")
    st.bar_chart(profile_df.set_index('Metric'))

    # ==============================
    # INSIGHTS
    # ==============================
    st.subheader("🧠 Personalized Insights")

    def explain(name, val):
        if val > 80:
            return f"{name} is excellent."
        elif val > 60:
            return f"{name} is above average."
        elif val > 40:
            return f"{name} is average."
        elif val > 20:
            return f"{name} is below average."
        else:
            return f"{name} is poor and needs attention."

    st.write(f"""
    • Sleep: {explain("Sleep quality", sleep_rank)}  
    • Activity: {explain("Activity level", steps_rank)}  
    • Hydration: {explain("Hydration", water_rank)}  
    • Heart Health: {explain("Heart condition", hr_rank)}  

    Overall, your profile suggests **{category.lower()} health status**.
    """)

    # ==============================
    # SUGGESTIONS
    # ==============================
    st.subheader("💡 Recommendations")

    if hr > 100:
        st.success("Improve cardiovascular fitness & reduce stress")

    if sleep < 6:
        st.success("Aim for 7–8 hours of sleep")

    if steps < 5000:
        st.success("Increase daily activity to 8,000+ steps")

    if water < 1.5:
        st.success("Increase hydration (2–3L daily)")

    if bmi > 25:
        st.success("Work on weight management through diet & exercise")
