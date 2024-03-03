import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Function to load your model from a pickle file
def load_model_lass():
    with open('your_lass_model.pkl', 'rb') as f:
        model_lass = pickle.load(f)
    return model_lass
def load_model_lin():
    with open('your_lin_model.pkl', 'rb') as f:
        model_lin = pickle.load(f)
    return model_lin
# Function to preprocess input data (replace this with your preprocessing code)
def preprocess_input(input_data):
    # Example: preprocessing steps such as scaling or encoding
    return input_data

# Function to make prediction
def predict(model, input_data):
    input_data = np.array([input_data])  # Convert input data to 2D array
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title('second hand car price Prediction ')

    # Input fields for 7 features
    st.write("Enter the car features")
    feature1 = st.number_input('year of launch',value=0)
    feature2 = st.number_input('orginal price(in lakhs)', value=0.0)
    feature3 = st.number_input('no. of kms travelled', value=0)
    st.write('0 : petrol')
    st.write('1 : disel')
    st.write('2 : CNG')
    feature4 = st.number_input('fuel type', value=0)
    #Seller_Type Transmission  Owner
    st.write('0 : dealer')
    st.write('1 : individual')
    feature5 = st.number_input('seller type', value=0)
    st.write('0 : manual')
    st.write('1 : automatic')
    feature6 = st.number_input('gear system', value=0)
    feature7 = st.number_input('no. of owners', value=0)

    input_data = [feature1, feature2, feature3, feature4, feature5, feature6, feature7]

    if st.button('Predict'):
        # Load the model
        model_lass = load_model_lass()
        model_lin = load_model_lin()
        # Preprocess the input data
        input_data_processed = preprocess_input(input_data)

        # Make prediction
        prediction_lass = predict(model_lass, input_data_processed)
        st.write('prediction of lasso regression model')
        st.success(f'Predicted price: {prediction_lass[0].round(3)} lakhs')
        prediction_lin = predict(model_lin, input_data_processed)
        st.write('prediction of linear regression model')
        st.success(f'Predicted price: {prediction_lin[0].round(3)} lakhs')
        st.write('we can observe the difference between the predictions of both models')
        st.write("NOTE: the above prediction values are close to actual value...! ")

if __name__ == '__main__':
    main()
