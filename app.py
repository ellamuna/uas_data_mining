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

data = data[['diagnosis','radius_mean','area_mean', 'radius_se', 'area_se', 'smoothness_mean','smoothness_se']]

st.title('Aplikasi Prediksi Kanker Payudara')

html_layout1 = """
<br>
<div style="background-color:#474747 ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)

activities = ['Logistic Regresion']
option = st.sidebar.selectbox('Pilih metode ?',activities)

st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Breast Cancer</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data, explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#mengubah B dan M pada kolom diagnosis menjadi 1 dan 0
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)

#train test split
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

param_range = np.linspace(0.1, 1.0, 5)


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
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)
    plt.grid()
    plt.legend(loc="best")


def user_report():
    radius_mean = st.sidebar.slider('Radius Mean', 0.0, 10.0, 30.0)
    area_mean = st.sidebar.slider('Area Mean', 0.0, 100.0, 2501.0)
    radius_se = st.sidebar.slider('Radius Se', 0.0, 1.0, 2.8)
    area_se = st.sidebar.slider('Area Se', 0.0, 50.0, 542.2)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.00001, 0.1090, 0.1634)
    smoothness_se = st.sidebar.slider('Smoothnes Se', 0.00001, 0.0060, 0.0310 )

    
    user_report_data = {
        'radius_mean':radius_mean,
        'area_mean': area_mean,
        'radius_se': radius_se,
        'area_se':area_se,
        'smoothness_mean':smoothness_mean,
        'smoothness_se':smoothness_se
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
    output='Kamu terkena kanker ganas'
else:
    output ='Kamu terkena kanker jinak'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(lr_score*100)+'%')