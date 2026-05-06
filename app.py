import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('house_price_model.pkl')

# App title
st.title('🏠 California House Price Predictor')
st.write('Enter the details below to predict the house price.')

# Input fields
st.sidebar.header('Input Features')

med_inc = st.sidebar.slider('Median Income ($10,000s)', 0.5, 15.0, 5.0, 0.1)
house_age = st.sidebar.slider('House Age (years)', 1, 52, 20, 1)
ave_rooms = st.sidebar.slider('Average Rooms', 1.0, 10.0, 5.0, 0.1)
ave_occup = st.sidebar.slider('Average Occupancy', 1.0, 10.0, 3.0, 0.1)
latitude = st.sidebar.slider('Latitude', 32.5, 42.0, 35.0, 0.1)
longitude = st.sidebar.slider('Longitude', -124.5, -114.0, -119.0, 0.1)

# Predict button
if st.button('Predict Price'):
    features = np.array([[med_inc, house_age, ave_rooms, ave_occup, latitude, longitude]])
    prediction = model.predict(features)[0]
    price = prediction * 100000
    
    st.success(f'### Predicted House Price: ${price:,.0f}')
    
    # Show input summary
    st.subheader('Input Summary')
    st.write(f'- Median Income: ${med_inc * 10000:,.0f}')
    st.write(f'- House Age: {house_age} years')
    st.write(f'- Average Rooms: {ave_rooms}')
    st.write(f'- Average Occupancy: {ave_occup}')
    st.write(f'- Location: ({latitude}, {longitude})')

# Model info
st.markdown('---')
st.subheader('Model Performance')
col1, col2, col3 = st.columns(3)
col1.metric('R² Score', '0.582')
col2.metric('RMSE', '$73,983')
col3.metric('MAE', '$54,257')