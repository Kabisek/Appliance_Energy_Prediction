import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Set page config
st.set_page_config(page_title="Appliances Energy Consumption Prediction", layout="wide")

# Title and description
st.title("📊 Appliances Energy Consumption Prediction")

# Load data
@st.cache_data
def load_data():
    data_path = "data/processed/combined_data.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please ensure the file exists.")
        st.stop()
    data = pd.read_csv(data_path)
    return data


df = load_data()


# Load LSTM model
@st.cache_resource
def load_lstm_model():
    model_path = "models/lstm_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure the file exists.")
        st.stop()
    model = load_model(model_path)
    return model


model = load_lstm_model()

# Feature selection
features = st.multiselect("Select features for the model:",
                          [col for col in df.columns if col != "Appliances"],
                          default=[col for col in df.columns if col != "Appliances"])

if features:
    # Prepare scaler
    scaler = MinMaxScaler()
    scaler.fit(df[features + ["Appliances"]])

    # Interactive prediction
    st.subheader("Make a Prediction")
    st.markdown("Enter values for each feature.")

    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}:",
                                              value=float(df[feature].mean()),
                                              key=feature)

    if st.button("Predict"):
        # Prepare input for LSTM
        input_df = pd.DataFrame([input_data], columns=features)

        # Scale input
        dummy_array = np.zeros((1, len(features) + 1))
        dummy_array[:, :-1] = input_df[features].values
        scaled_input = scaler.transform(dummy_array)
        scaled_input = scaled_input[:, :-1].reshape(1, 1, len(features))

        # Predict
        prediction = model.predict(scaled_input)

        # Inverse scale the prediction
        dummy_array = np.zeros((1, len(features) + 1))
        dummy_array[:, -1] = prediction
        prediction_unscaled = scaler.inverse_transform(dummy_array)[:, -1][0]

        # Display prediction with unit
        st.write(f"Predicted Appliances Energy Consumption: {prediction_unscaled:.2f} Watt-hour")

# Footer
st.markdown("---")
