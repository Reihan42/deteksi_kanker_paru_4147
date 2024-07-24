import streamlit as st
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib

nb_model = joblib.load("NB.pkl")
svm_model = joblib.load("SVM.pkl")

dataset = pd.read_csv("lung cancer.csv")

# Evaluating model accuracy
X = dataset[["Age",
 "Gender",
 "Air Pollution",
 "Alcohol use",
 "OccuPational Hazards",
 "Genetic Risk",
 "Balanced Diet",
 "Obesity",
 "Smoking",
 "Passive Smoker",
 "Chest Pain",
 "Coughing of Blood",
 "Fatigue",
 "Weight Loss",
 "Shortness of Breath",
 "Swallowing Difficulty",
 "Clubbing of Finger Nails",
 "Dry Cough",
 "Snoring"]]
y = dataset["Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Naive Bayes accuracy
nb_test_pred = nb_model.predict(X_test)
nb_test_accuracy = accuracy_score(y_test, nb_test_pred)

# SVM accuracy
svm_test_pred = svm_model.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)

def get_unique_values(column_name):
    return dataset[column_name].unique()

st.title("Prediksi Penyakit Kanker Paru Paru")
st.text("""Merupakan sebuah alat prediksi yang dikhususkan untuk alhi (dokter) 
menggunakan algoritma Naive Bayes dan Algortima SVM""")
st.markdown(f"Akurasi **{nb_test_accuracy * 100:.2f}%** untuk Naive Bayes & **{svm_test_accuracy * 100:.2f}%** untuk SVM.")
gender_options = {"Male": 1, "Female": 2}
Option_1_to_7 = {"Very Low": 1,"Low": 2,"Low Medium": 3,"Medium": 4,"Medium High": 5,"High": 6,"Very High": 7}
Option_1_to_8 = {"Lowest": 1,"Very Low": 2,"Low": 3,"Low Medium": 4,"Medium": 5,"Medium High": 6,"High": 7,"Very High": 8}
Option_1_to_9 = {"Lowest": 1,"Very Low": 2,"Low": 3,"Low Medium": 4,"Medium": 5,"Medium High": 6,"High": 7,"Very High": 8, "Highest":9}


def user_input_feature():
    age = st.number_input("Masukkan umur", min_value=0, value=0, placeholder="Type a number...")
    selected_gender = st.radio("Pilih jenis kelamin:", options=list(gender_options.keys()))
    air_pollution_option = st.select_slider("Seberapa tinggi polusi udara pasien:", options=list(Option_1_to_8.keys()))
    alcohol_use_option = st.select_slider("Seberapa tinggi konsumsi minuman keras pasien:", options=list(Option_1_to_8.keys()))
    occu_hazard_option = st.select_slider("Seberapa tinggi tingkat bahaya pekerjaan pasien:", options=list(Option_1_to_8.keys()))
    gen_risk_option = st.select_slider("Seberapa tinggi riwayat penyakit keluarga/genetik pasien:", options=list(Option_1_to_7.keys()))
    bal_diet_option = st.select_slider("Seberapa tinggi diet seimbang pasien:", options=list(Option_1_to_7.keys()))
    obesity_option = st.select_slider("Seberapa tinggi tingkat obesitas pasien:", options=list(Option_1_to_7.keys()))
    smoking_option = st.select_slider("Seberapa tinggi tingkat merokok", options=list(Option_1_to_8.keys()))
    passive_smoker_option = st.select_slider("Seberapa tinggi tingkat lingkungan asap rokok disektiar pasien:", options=list(Option_1_to_8.keys()))
    chest_pain_option = st.select_slider("Seberapa tinggi rasa sakit pasien dibagian dada:", options=list(Option_1_to_9.keys()))
    cough_blood_option = st.select_slider("Seberapa tinggi tingkat keseringan pasien batuk berdarah :", options=list(Option_1_to_9.keys()))
    fatigue_option = st.select_slider("Seberapa tinggi rasa lelah pasien:", options=list(Option_1_to_9.keys()))
    weight_loss_option = st.select_slider("Seberapa tinggi tingkat pasien kehilangan berat badan:", options=list(Option_1_to_8.keys()))
    short_breath_option = st.select_slider("Seberapa tinggi tingkat kependekan nafas pasien", options=list(Option_1_to_9.keys()))
    swallow_diff_option = st.select_slider("Seberapa tinggi tingkat kesulitan menelan pasien:", options=list(Option_1_to_8.keys()))
    clubbing_nails_option = st.select_slider("Seberapa tinggi tingkat kebengkakan jari pasien:", options=list(Option_1_to_9.keys()))
    dry_cough_option = st.select_slider("Seberapa tinggi tingkat batuk kering pasien:", options=list(Option_1_to_7.keys()))
    snoring_option = st.select_slider("Seberapa tinggi tingkat keseringan pasien mendengkur:", options=list(Option_1_to_7.keys()))
    gender = gender_options[selected_gender]
    air_pollution = Option_1_to_8[air_pollution_option]
    alcohol_use = Option_1_to_8[alcohol_use_option]
    occu_hazard = Option_1_to_8[occu_hazard_option]
    gen_risk = Option_1_to_7[gen_risk_option]
    bal_diet = Option_1_to_7[bal_diet_option]
    obesity = Option_1_to_7[obesity_option]
    smoking = Option_1_to_8[smoking_option]
    passive_smoker = Option_1_to_8[passive_smoker_option]
    chest_pain = Option_1_to_9[chest_pain_option]
    cough_blood = Option_1_to_9[cough_blood_option]
    fatigue = Option_1_to_9[fatigue_option]
    weight_loss = Option_1_to_8[weight_loss_option]
    short_breath = Option_1_to_9[short_breath_option]
    swallow_diff = Option_1_to_8[swallow_diff_option]
    clubbing_nails = Option_1_to_9[clubbing_nails_option]
    dry_cough = Option_1_to_7[dry_cough_option]
    snoring = Option_1_to_7[snoring_option]

    data = {"Age": age,
            "Gender": gender,
            "Air Pollution": air_pollution,
            "Alcohol use": alcohol_use,
            "OccuPational Hazards": occu_hazard,
            "Genetic Risk": gen_risk,
            "Balanced Diet": bal_diet,
            "Obesity": obesity,
            "Smoking": smoking,
            "Passive Smoker": passive_smoker,
            "Chest Pain": chest_pain,
            "Coughing of Blood": cough_blood,
            "Fatigue": fatigue,
            "Weight Loss": weight_loss,
            "Shortness of Breath": short_breath,
            "Swallowing Difficulty": swallow_diff,
            "Clubbing of Finger Nails": clubbing_nails,
            "Dry Cough": dry_cough,
            "Snoring": snoring}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_feature()

lung_cancer_raw = pd.read_csv("lung cancer.csv")

lung_cancer = lung_cancer_raw[["Age",
                                "Gender",
                                "Air Pollution",
                                "Alcohol use",
                                "OccuPational Hazards",
                                "Genetic Risk",
                                "Balanced Diet",
                                "Obesity",
                                "Smoking",
                                "Passive Smoker",
                                "Chest Pain",
                                "Coughing of Blood",
                                "Fatigue",
                                "Weight Loss",
                                "Shortness of Breath",
                                "Swallowing Difficulty",
                                "Clubbing of Finger Nails",
                                "Dry Cough",
                                "Snoring"]]

df = pd.concat([input_df,lung_cancer], axis=0)
df = df[:1]

st.subheader("User Input")
st.write(df)


def nb_pred(input_data):
    prediction = nb_model.predict(input_data)
    return prediction

def predict_svm(input_data):
    prediction = svm_model.predict(input_data)
    return prediction

if st.button("Prediksi dengan Naive Bayes"):
    nb_prediction = nb_pred(df)
    st.write(f"Level kanker: {nb_prediction[0]}")

if st.button("Prediksi dengan Support Vector Machine"):
    svm_prediction = predict_svm(df)
    st.write(f"Level kanker: {svm_prediction[0]}")
