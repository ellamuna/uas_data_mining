import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pickle 

#import model 
lr = pickle.load(open('lr.pkl','rb'))

#load dataset
data = pd.read_csv('Breast Cancer.csv')

st.title('Breast Cancer Application')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['LR','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')


if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset PIMA Indian</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X=data.drop(['diagnosis'],axis=1)
y = data['diagnosis']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    kehamilan = st.sidebar.slider('Kehamilan',0,20,1)
    glukosa = st.sidebar.slider('Glukosa',0,200,108)
    bp = st.sidebar.slider('Tekanan Darah',0,140,40)
    skinthickness = st.sidebar.slider('Ketebalan Kulit',0,100,25)
    insulin = st.sidebar.slider('Insulin',0,1000,120)
    bmi = st.sidebar.slider('BMI',0,80,25)
    diabetespd = st.sidebar.slider('Diabetes Pedigree', 0.05,2.5,0.45)
    age = st.sidebar.slider('Usia',21,100,24)
    
    user_report_data = {
        'Pregnancies':kehamilan,
        'Glucose':glukosa,
        'BloodPressure':bp,
        'SkinThickness':skinthickness,
        'Insulin':insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction':diabetespd,
        'Age':age
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = lr.predict(user_data)
lr_score = accuracy_score(y_test,lr.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena diabetes'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(lr_score*100)+'%')