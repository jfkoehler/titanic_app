import streamlit as st 
import pickle
import numpy as np
import pandas as pd

st.header("Predict Cruise Survival")

st.write("This model predicts whether you will survive your next boat trip.")

with open('forest.pkl', 'rb') as f:
    model = pickle.load(f)

sex = st.radio('Sex', ['male', 'female'])
embark_town = st.selectbox('Embarking From', ['Southampton', 'Cherbourg', 'Queenstown'])
age = st.slider("Age", min_value = 0, max_value = 100)
fare = st.slider('Fare', min_value = 0, max_value = 500)

X = np.array([[sex, age, fare, embark_town]])
X = pd.DataFrame({'sex': [sex], 'age': [age], 'fare': [fare],
                    'embark_town': [embark_town]})
prediction = model.predict(X)
pred_dict = {1: 'survived', 0: 'die!'}
st.write(f'Your prediction is {pred_dict[prediction[0]]}')