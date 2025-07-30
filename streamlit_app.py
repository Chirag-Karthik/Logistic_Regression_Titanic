import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained logistic regression model
# Make sure 'logistic_regression_model.joblib' is in the same directory as this script
try:
    logreg = joblib.load('logistic_regression_model.joblib')
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.joblib' not found. Please make sure it's in the same directory.")
    st.stop() # Stop the app if the model file is not found

st.title('Titanic Survival Prediction')
st.write('Enter the passenger details to predict their survival probability on the Titanic.')

# Create input fields for each feature
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 30)
sibsp = st.slider('Number of Siblings/Spouses Aboard (SibSp)', 0, 8, 0)
parch = st.slider('Number of Parents/Children Aboard (Parch)', 0, 6, 0)
fare = st.number_input('Fare', value=30.0)
embarked = st.selectbox('Port of Embarkation (Embarked)', ['C', 'Q', 'S'])

# Map categorical inputs to numerical values used during training
# Ensure these mappings match the LabelEncoding used in the notebook
sex_mapping = {'male': 1, 'female': 0}
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2} # Assuming this was the encoding order

sex_encoded = sex_mapping[sex]
embarked_encoded = embarked_mapping[embarked]


# Create a button to make predictions
if st.button('Predict Survival'):
    # Prepare the input data as a pandas DataFrame
    # Ensure the column order matches the training data
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                               columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Make prediction
    prediction = logreg.predict(input_data)
    prediction_proba = logreg.predict_proba(input_data)[:, 1] # Probability of survival

    # Display the prediction result
    if prediction[0] == 1:
        st.success(f'The model predicts that this passenger **survived** with a probability of {prediction_proba[0]:.2f}.')
    else:
        st.error(f'The model predicts that this passenger **did not survive** with a probability of {prediction_proba[0]:.2f}.')

st.write('Note: This is a simplified model for demonstration purposes.')
