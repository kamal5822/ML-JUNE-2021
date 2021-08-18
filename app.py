import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('model_pickle','rb') as f:
    model = pickle.load(f)

columns = st.beta_columns((1,1))

def user_report():
  pregnancies = st.sidebar.slider('Pregnancies',min_value = 0 , max_value = 17 , value = 3 , step = 1)
  glucose = st.sidebar.slider('Glucose', min_value = 0 , max_value = 200 , value = 120 , step = 1)
  bp = st.sidebar.slider('Blood Pressure', min_value = 0 , max_value = 130 , value = 70 , step = 1)
  skinthickness = st.sidebar.slider('Skin Thickness', min_value = 0 , max_value = 100 , value = 21 , step = 1)
  insulin = st.sidebar.slider('Insulin', min_value = 0 , max_value = 850 , value = 80 , step = 1)
  bmi = st.sidebar.slider('BMI', min_value = 0.0 , max_value = 70.0 , value = 32.0 , step = 0.1)
  dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value = 0.000 , max_value = 3.000 , value = 0.471 , step = 0.001 )
  age = st.sidebar.slider('Age',min_value = 21 , max_value = 100 , value = 34 , step = 1)

  with columns[0]:
    st.subheader('Data Entered')
    st.subheader(' ')
    st.write('Pregnancies : ',pregnancies)
    st.write('Glucose : ',glucose)
    st.write('BloodPressure : ',bp)
    st.write('SkinThickness : ',skinthickness)
    
  with columns[1]:
    st.subheader(' ')
    st.subheader(' ')
    st.write(' ')
    st.write('Insulin : ',insulin)
    st.write('BMI : ',bmi)
    st.write('Diabeted Pedigree Function : ',dpf)
    st.write('Age : ',age)



  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()

input_data = user_data.iloc[:,:].values

ss = StandardScaler()

df = pd.read_csv('diabetes.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

z = ss.transform(input_data)

user_result = model.predict(z)

st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'You are healthy'
else:
    output = 'You are diabetic'

st.write(output)

st.subheader('Data Used Details')

Description = st.beta_expander("Data Description",expanded = False)
with Description:
  st.write(df.describe())

image = Image.open('Capture.JPG')

Visualization = st.beta_expander("Data Barplot",expanded = False)
with Visualization:
  st.image(image,use_column_width = True)

y_pred = model.predict(x_test)

Accuracy = st.beta_expander("Data Accuracy Score",expanded = False)
with Accuracy:
  st.write('Accuracy Score : ',accuracy_score(y_pred,y_test))


