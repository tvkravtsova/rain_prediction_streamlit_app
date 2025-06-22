import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🌧️ Rain Forecast in Australia")

# Info section
st.write("---")
st.write("### ℹ️ About This App")
st.write("""
This app uses machine learning to predict rain.
The model is trained on Australian weather data.

**How to use:**
1. Enter weather data in the form
2. Click the 'Predict Rain' button
3. See the result and prediction confidence

**Model accuracy:** approximately 85%
""")
# Show map
st.image("images/Australian-climate-zones.png", caption="Climate zones of Australia", use_column_width=True)

st.write("Enter weather data to predict whether it will rain tomorrow.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/aussie_rain.joblib')

# Load model and components
model_data = load_model()
model = model_data['model']
imputer = model_data['imputer']
scaler = model_data['scaler']
encoder = model_data['encoder']
numeric_cols = model_data['numeric_cols']
categorical_cols = model_data['categorical_cols']
input_cols = model_data['input_cols']
categories = model_data.get('categories', {})

# Form UI
st.header("📝 Enter Weather Data:")

with st.form("weather_form"):
    input_data = {}
    col1, col2 = st.columns(2)

    for i, col in enumerate(input_cols):
        target_col = col1 if i % 2 == 0 else col2
        if col in numeric_cols:
            input_data[col] = target_col.number_input(f"{col}:", value=0.0)
        else:
            options = categories.get(col)
            if options:
                input_data[col] = target_col.selectbox(f"{col}:", sorted(options))
            else:
                input_data[col] = target_col.text_input(f"{col}:", value="")

    predict_button = st.form_submit_button("🔮 Predict Rain")

# Prediction logic
if predict_button:
    try:
        input_df = pd.DataFrame([input_data])

        for col in categorical_cols:
            if input_df[col].iloc[0] == "":
                input_df[col] = "Unknown"

        input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        input_encoded = encoder.transform(input_df[categorical_cols])
        input_final = np.hstack([input_df[numeric_cols].values, input_encoded])

        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]

        st.header("🎯 Prediction Result:")
        if prediction == "Yes":
            st.success("🌧️ IT WILL RAIN TOMORROW!")
            prob = probability[model.classes_.tolist().index("Yes")]
            st.write(f"Rain probability: {prob:.1%}")
        else:
            st.info("☀️ No rain tomorrow")
            prob = probability[model.classes_.tolist().index("No")]
            st.write(f"No-rain probability: {prob:.1%}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check your input values.")

       

