import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Load model and encoder
# ---------------------------------------------------
model = joblib.load("src/final_model.pkl")
label_encoder = joblib.load("src/label_encoder.pkl")

# Get expected raw input columns from fitted preprocessor
preprocessor = model.named_steps["preprocessor"]
expected_columns = list(preprocessor.feature_names_in_)

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="Rain Prediction App", page_icon="🌧️")

st.title("🌧️ RainTomorrow Prediction App")
st.write("Enter weather conditions to predict whether it will rain tomorrow.")

st.subheader("Model expects these input columns:")
st.write(expected_columns)

# ---------------------------------------------------
# User inputs
# ---------------------------------------------------
date_value = st.date_input("Date")
date_value = str(date_value)

min_temp = st.number_input("Min Temperature", value=10.0)
max_temp = st.number_input("Max Temperature", value=20.0)
rainfall = st.number_input("Rainfall", value=0.0)
evaporation = st.number_input("Evaporation", value=5.0)
sunshine = st.number_input("Sunshine", value=7.0)
wind_gust_speed = st.number_input("Wind Gust Speed", value=30.0)
humidity_9am = st.number_input("Humidity at 9am", value=60.0)
humidity_3pm = st.number_input("Humidity at 3pm", value=50.0)
pressure_9am = st.number_input("Pressure at 9am", value=1015.0)
pressure_3pm = st.number_input("Pressure at 3pm", value=1012.0)
temp_9am = st.number_input("Temperature at 9am", value=15.0)
temp_3pm = st.number_input("Temperature at 3pm", value=19.0)
cloud_9am = st.number_input("Cloud at 9am", value=4.0)
cloud_3pm = st.number_input("Cloud at 3pm", value=5.0)
wind_speed_9am = st.number_input("Wind Speed at 9am", value=15.0)
wind_speed_3pm = st.number_input("Wind Speed at 3pm", value=20.0)

location = st.selectbox(
    "Location",
    ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
)

wind_gust_dir = st.selectbox(
    "Wind Gust Direction",
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

wind_dir_9am = st.selectbox(
    "Wind Direction at 9am",
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

wind_dir_3pm = st.selectbox(
    "Wind Direction at 3pm",
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])

# Feature engineering used in notebook
temp_range = max_temp - min_temp

# ---------------------------------------------------
# Build a complete input row with ALL expected columns
# ---------------------------------------------------
default_values = {col: 0 for col in expected_columns}

# Reasonable defaults for categorical columns often present
categorical_defaults = {
    "Date": date_value,
    "Location": location,
    "WindGustDir": wind_gust_dir,
    "WindDir9am": wind_dir_9am,
    "WindDir3pm": wind_dir_3pm,
    "RainToday": rain_today,
}

numeric_updates = {
    "MinTemp": min_temp,
    "MaxTemp": max_temp,
    "Rainfall": rainfall,
    "Evaporation": evaporation,
    "Sunshine": sunshine,
    "WindGustSpeed": wind_gust_speed,
    "Humidity9am": humidity_9am,
    "Humidity3pm": humidity_3pm,
    "Pressure9am": pressure_9am,
    "Pressure3pm": pressure_3pm,
    "Temp9am": temp_9am,
    "Temp3pm": temp_3pm,
    "Cloud9am": cloud_9am,
    "Cloud3pm": cloud_3pm,
    "WindSpeed9am": wind_speed_9am,
    "WindSpeed3pm": wind_speed_3pm,
    "TempRange": temp_range,
}

for key, value in categorical_defaults.items():
    if key in default_values:
        default_values[key] = value

for key, value in numeric_updates.items():
    if key in default_values:
        default_values[key] = value

input_data = pd.DataFrame([default_values], columns=expected_columns)

# ---------------------------------------------------
# Show input data
# ---------------------------------------------------
st.subheader("Input Data Sent to Model")
st.dataframe(input_data)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Rain Tomorrow"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        prediction_proba = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write(f"**Rain Tomorrow:** {prediction_label}")
        st.write(f"**Probability of Rain:** {prediction_proba:.2%}")

        if prediction_label == "Yes":
            st.warning("It is likely to rain tomorrow.")
        else:
            st.success("It is unlikely to rain tomorrow.")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)