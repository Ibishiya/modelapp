import streamlit as st
import pickle
import numpy as np

# Load the model
with open(r'model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit front-end
st.title("Housing Price Prediction")

# Create input fields
square_footage = st.number_input("Enter square footage:")
bedrooms = st.number_input("Enter number of bedrooms:")
bathrooms = st.number_input("Enter number of bathrooms:")

# Button to make prediction
if st.button("Predict"):
    input_features = np.array([[square_footage, bedrooms, bathrooms]])
    prediction = model.predict(input_features)
    st.write("Predicted Price:", prediction[0])

# Additional features like charts, etc.
