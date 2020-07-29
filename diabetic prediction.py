import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 

st.write("""
# DIABETES PREDECTATION
Detect if someone has diabetes using machine learning with python!
""")
image=Image.open('C:/Users/user/Desktop/vartual assistant/download.jpg')
st.image(image,caption='Combination of ML and medicle science',use_column_width=True)

df=pd.read_csv('Desktop/datasets_228_482_diabetes (1).csv')

st.subheader('Data Information:')
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)

X=df.iloc[:, 0:8].values
Y=df.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

def get_user_input():
    Pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    Glucose=st.sidebar.slider('Glucose',0,199,117)
    Blood_Pressure=st.sidebar.slider('Blood_Pressure',0,122,72)
    Skin_Thickness=st.sidebar.slider('Skin_Thickness',0,99,23)
    Insulin=st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
    Diabetes_Pedigree_Function=st.sidebar.slider('Diabetes_Pedigree_Function',0.078,2.42,0.3725)
    Age=st.sidebar.slider('Age',21,81,29)

    user_data={ 'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'Blood_Pressure': Blood_Pressure,
                'Skin_Thickness': Skin_Thickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'Diabetes_Pedigree_Function': Diabetes_Pedigree_Function,
                'Age': Age,

            }

    features = pd.DataFrame(user_data, index= [0])
    return features

user_input = get_user_input() 

st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

st.subheader('Test Accuracy score By Modle:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

prediction=RandomForestClassifier.predict(user_input)

st.subheader('Predicted output:')
st.write(prediction)
