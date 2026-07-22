import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load The Trained Model
model = tf.keras.models.load_model('ann_model.h5')

## load the scaler and encoders
with open('label_encode_gender.pkl', 'rb') as file:
   label_encode_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



##Streamlit app
st.title("Customer Churn Prediction")
##USER INPUT
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st. selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = one_hot_encoder.transform(
    input_data[['Geography']]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.drop('Geography', axis=1), geo_encoded_df],
    axis=1
)



# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(f"CHhurn Probability :{prediction_proba:.2f}")

if prediction_proba > 0.5:
  st.write('The customer is likely to churn.')
else:
  st.write('The customer is.not likely.to churn.')