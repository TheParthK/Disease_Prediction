import string
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data = pd.read_csv('Dataset/Training.csv').dropna(axis = 1)
data["prognosis"] = encoder.fit_transform(data["prognosis"])

tuple_symptoms = (i for i in data.head())
st.title("Disease Prediction")
# option = st.multiselect(
#     'Select your symptoms',
#     ('Email', 'Home phone', 'Mobile phone'))
options = st.multiselect(
    'Select your symptoms',
    tuple_symptoms
    )

# st.write('You selected:', options)


loaded_model = pickle.load(open('dp_model_final.pkl', 'rb'))

X = data.iloc[:,:-1]

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][loaded_model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    # final_prediction = mode([rf_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        # "naive_bayes_prediction": nb_prediction,
        # "svm_model_prediction": nb_prediction,
        # "final_prediction":final_prediction
    }
    return predictions

# Testing the function

# print(predictDisease("Cough,Dark Urine,Pain In Anal Region,Weight Loss"))

# print(data_dict["symptom_index"])

def formattingText(text):
    text = text.split('_')
    text = map(str.title, text)
    return ' '.join(text)

isClicked = False


string_options = ''

predict_button = st.button(label='Predict')

if options and predict_button:
    predict_button = False
    string_options = ','.join(map(formattingText, options))
    st.write(predictDisease(string_options)["rf_model_prediction"])
# def ButtonClicked(activated=False):
#     if options and activated:
#         string_options = ','.join(map(formattingText, options))
#         st.write(predictDisease(string_options)["rf_model_prediction"])
#         # print(string_options)
#         # print(options)





# print(list(map(formattingText, options)))



