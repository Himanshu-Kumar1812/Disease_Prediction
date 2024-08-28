import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

# creating a function for prediction
def diabetes_prediction(input_data):
    

    #changing the input data to numpy array
    input_data_as_numpyarray = np.asarray(input_data)

    #reshape the array as we prediciting only 1 instance of the data
    input_data_reshaped = input_data_as_numpyarray.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "women is non-diabetic"
    else:
       return "women is diabetic"


def main():
    st.title("Diabetes Prediction..")

    #getting the input data from user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure value")
    SkinThickness = st.text_input("SkinThickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")


    # code for prediction
    diagnosis = ''
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

    


if __name__ == '__main__':
    main()
