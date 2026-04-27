import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Health AI", layout="centered")

# ==============================
# MOBILE STYLE CSS
# ==============================
st.markdown("""
<style>
.block-container {
    max-width: 420px;
    padding-top: 1rem;
}

.card {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.title {
    font-size: 22px;
    font-weight: 600;
}

.subtitle {
    color: grey;
    font-size: 14px;
}

.center {
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
# HEADER
# ==============================
st.markdown('<div class="title center">🧠 Health AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle center">Quick health check</div>', unsafe_allow_html=True)

# ==============================
# INPUT CARD
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

age = st.number_input("Age", 10, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
sex = 1 if gender == "Male" else 0

bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

hr = st.number_input("Heart Rate", 40, 180, 75)
steps = st.number_input("Steps", 0, 20000, 8000)
sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
water = st.number_input("Water Intake (L)", 0.0, 5.0, 2.0)

analyze = st.button("Analyze")

st.markdown('</div>', unsafe_allow_html=True)

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
if analyze:

    similar_users = get_similar_users(df_user, age, sex, bmi, smoking)
    if len(similar_users) < 10:
        similar_users = df_user

    avg_vals = similar_users.mean(numeric_only=True)

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
    # MODEL
    # ==============================
    X_scaled = scaler.transform(new_user[features])
    prob = model.predict_proba(X_scaled)[0][1]
    score = 100 - prob*100

    # ==============================
    # RESULT CARD
    # ==============================
    st.markdown('<div class="card center">', unsafe_allow_html=True)

    st.markdown(f"<h2>{round(score,1)}</h2>", unsafe_allow_html=True)
    st.markdown("Health Score")

    if prob < 0.3:
        st.success("Healthy")
    elif prob < 0.7:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # GAUGE
    # ==============================
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4d4d"},
                {'range': [40, 70], 'color': "#ffa64d"},
                {'range': [70, 100], 'color': "#66cc66"}
            ]
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # ==============================
    # RADAR
    # ==============================
    radar_data = {
        "Sleep": sleep/10,
        "Steps": steps/10000,
        "Water": water/3,
        "Heart": 1-(hr/150),
        "BMI": 1-(bmi/40)
    }

    categories = list(radar_data.keys())
    values = list(radar_data.values())
    values += values[:1]
    categories += categories[:1]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,1])))

    st.plotly_chart(fig_radar, use_container_width=True)

    # ==============================
    # INSIGHTS
    # ==============================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🧠 Insights")

    if sleep < 6:
        st.write("• Your sleep is lower than recommended.")
    if steps < 5000:
        st.write("• Activity level is low compared to others.")
    if water < 1.5:
        st.write("• Hydration is insufficient.")
    if hr > 100:
        st.write("• Heart rate is elevated.")

    if sleep >=6 and steps>=5000 and water>=1.5:
        st.write("• Your overall lifestyle looks balanced.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # SAVE DATA
    # ==============================
    new_user.to_csv("new_entry.csv", mode='a', header=False, index=False)
