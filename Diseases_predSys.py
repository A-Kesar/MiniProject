

import pickle
import streamlit as st    
from streamlit_option_menu import option_menu
import numpy as np


# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('Heart_model.sav','rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3= st.columns(3)
    
    with col1:
        Pregnancies =  st.number_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.number_input('Glucose Level')
    
    with col3:
        BloodPressure = st.number_input('Enter Diastolic Blood Pressure')
    
    with col1:
        BMI = st.number_input('BMI value')
    
    with col2:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col3:
       Age = st.number_input('Age of the Person')

    
    with open('max_min', 'rb') as f:
        min_max_list= pickle.load(f)

    Pregnancies=(Pregnancies-min_max_list[0][0])/(min_max_list[0][1]-min_max_list[0][0])
    Glucose=(Glucose-min_max_list[1][0])/(min_max_list[1][1]-min_max_list[1][0])
    BloodPressure=(BloodPressure-min_max_list[2][0])/(min_max_list[2][1]-min_max_list[2][0])
    BMI=(BMI-min_max_list[3][0])/(min_max_list[3][1]-min_max_list[3][0])
    DiabetesPedigreeFunction=(DiabetesPedigreeFunction-min_max_list[4][0])/(min_max_list[4][1]-min_max_list[2][0])
    Age=(Age-min_max_list[5][0])/(min_max_list[5][1]-min_max_list[5][0])

    input_data=[Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(input_data_reshaped)
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex: Male=0 , Female=1')
        
    with col3:
        cp = st.number_input('Chest Pain : Asymptomatic=0,Not Angina=1,Angina=2,Abnormal=3')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure <94-200>')
        
    with col2:
        chol = st.number_input('Serum Cholestoral  (mg/dl) <126-564>')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl : True=1, False=0')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results: norm:0,abnormal:1,hyper:2')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved:<71-202>')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina -> Yes:1, No 0')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise :<0-6.2>')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment: Up=1,flat=2,down=3')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy:(0,1,2,3)')
        
    with col1:
        thal = st.number_input('thal: 1 = reversable defect; 2 = fixed defect; 3 = normal')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    




