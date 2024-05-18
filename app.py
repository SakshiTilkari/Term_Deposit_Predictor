import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model from the pickle file
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the input function
def user_input_features():
    st.header("User Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider('Age', 18, 100, 30)
        job = st.selectbox('Job', [
            'Admin', 'Blue-Collar', 'Entrepreneur', 'Housemaid', 'Management',
            'Retired', 'Self-Employed', 'Services', 'Student', 'Technician', 'Unemployed', 'Unknown'
        ])
        marital = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
        education = st.selectbox('Education', ['Primary', 'Secondary', 'Tertiary', 'Unknown'])
        default = st.selectbox('Default', ['No', 'Yes'])

    with col2:
        balance = st.slider('Balance', -5000, 49000, 0)
        housing = st.selectbox('Housing', ['No', 'Yes'])
        loan = st.selectbox('Loan', ['No', 'Yes'])
        contact = st.selectbox('Contact', ['Cellular', 'Telephone', 'Unknown'])
        day = st.slider('Day', 1, 31, 15)

    with col3:
        month = st.selectbox('Month', [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ])
        duration = st.slider('Duration', 0, 2500, 300)
        campaign = st.slider('Campaign', 1, 25, 1)
        pdays = st.slider('Pdays', -1, 500, 0)
        previous = st.slider('Previous', 0, 100, 0)
    poutcome = st.selectbox('Previous Outcome', ['Failure', 'Other', 'Success', 'Unknown'])
    
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


# Input Data

input_df = user_input_features()

# Display User Input

st.subheader('User Input Data')
st.write(input_df)

# Encode input data to match model's encoding
input_df_encoded = input_df.copy()
label_encoders = {}

for column in input_df.columns:
    if column in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        le = LabelEncoder()
        input_df_encoded[column] = le.fit_transform(input_df[column])
        label_encoders[column] = le

# Predict button
if st.button('Predict'):
    # Make predictions
    prediction = model.predict(input_df_encoded)
    prediction_proba = model.predict_proba(input_df_encoded)

    st.subheader('Prediction')
    st.write('Yes, the client will subscribe to a term deposit' if prediction[0] == 1 else "No, the client won't subscribe to a term deposit")

