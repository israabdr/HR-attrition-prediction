import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configure la page
st.set_page_config(
    page_title="Mon Tableau de Bord HR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le modèle
def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
svm_classifier_loaded = data["model"]
bt = data["bt"]
cf = data["cf"]
dp = data["dp"]
ef = data["ef"]
ge = data["ge"]
jr = data["jr"]
ms = data["ms"]
ot = data["ot"]
ed = data["ed"]

# Appliquer un style CSS pour rendre les titres en gras
def set_css():
    st.markdown(
        """
        <style>
        h1, h2, h3, h4, h5, h6 {
            font-weight: bold; /* Rendre tous les titres en gras */
        }
        .card {
            background-color: #EBACA2;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_css()  # Appliquer le style CSS global

# Fonction de prédiction
def predict(df):
    df["Business Travel"] = bt.transform(df["Business Travel"])
    df['CF_age band'] = df['CF_age band'].replace({'Under 25': 1, '25 - 34': 2, '35 - 44': 3, '45 - 54': 4, 'Over 55': 5})
    df["CF_attrition label"] = cf.transform(df["CF_attrition label"])
    df["Department"] = dp.transform(df["Department"])
    df["Education Field"] = ef.transform(df["Education Field"])
    df["Gender"] = ge.transform(df["Gender"])
    df["Job Role"] = jr.transform(df["Job Role"])
    df["Marital Status"] = ms.transform(df["Marital Status"])
    df["Over Time"] = ot.transform(df["Over Time"])
    df["Education"] = ed.transform(df["Education"])
    y_pred = svm_classifier_loaded.predict(df)
    return y_pred

# Fonction pour afficher le tableau de bord
def afficher_dashboard():
    df = pd.read_excel('Tableau HR Data.xlsx')
    total_employees = df.shape[0]
    total_ex_employees = df[df['Attrition'] == 'Yes'].shape[0]
    average_age = df['Age'].mean()
    attrition_rate = total_ex_employees / total_employees * 100

    st.sidebar.header("Filtres")
    st.subheader('HR DASHBOARD')

    # Filtres
    selected_department = st.sidebar.selectbox("Sélectionnez un département", df['Department'].unique())
    selected_education_field = st.sidebar.selectbox("Sélectionnez un domaine d'éducation", df['Education Field'].unique())

    with st.container():
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

        with row1_col1:
            st.markdown(f"<div class='card'><h4>Nombre total d'employés</h4><p>{total_employees}</p></div>", unsafe_allow_html=True)

        with row1_col2:
            st.markdown(f"<div class='card'><h4>Nombre total d'ex-employés</h4><p>{total_ex_employees}</p></div>", unsafe_allow_html=True)

        with row1_col3:
            st.markdown(f"<div class='card'><h4>Âge moyen</h4><p>{average_age:.2f} ans</p></div>", unsafe_allow_html=True)

        with row1_col4:
            st.markdown(f"<div class='card'><h4>Taux d'attrition</h4><p>{attrition_rate:.2f}%</p></div>", unsafe_allow_html=True)

    # Attrition par département
    with st.container():
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown(f"Attrition par département : {selected_department}")
            attrition_by_department = df[df['Department'] == selected_department]['Attrition'].value_counts()

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(attrition_by_department, labels=attrition_by_department.index, autopct='%1.1f%%', colors=['#CE6A6B', '#EBACA2'])
            ax.axis('equal')
            st.pyplot(fig)

        with row2_col2:
            st.markdown(f"Attrition par domaine d'éducation : {selected_education_field}")

            filtered_df = df[df['Education Field'] == selected_education_field]

            fig, ax = plt.subplots(figsize=(6, 4))
            filtered_df.groupby('Gender')['Attrition'].value_counts().unstack().plot(kind='bar', stacked=True, ax=ax, color=['#CE6A6B', '#EBACA2'])
            ax.set_xlabel('Genre')
            ax.set_ylabel("Nombre d'employés")
            ax.set_title(f'Attrition par Domaine d\'Éducation et Sexe ({selected_education_field})')
            ax.grid(True)
            st.pyplot(fig)

    # Satisfaction au travail par rôle professionnel
    with st.container():
        st.markdown("Satisfaction au Travail par Rôle Professionnel")
        cmap = LinearSegmentedColormap.from_list("custom_pink", ['#CE6A6B', '#EBACA2'])

        job_satisfaction_mean = df.groupby("Job Role")["Job Satisfaction"].mean().reset_index()
        pivot_table = job_satisfaction_mean.pivot_table(index="Job Role", values="Job Satisfaction")

        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(pivot_table, annot=True, cmap=cmap, ax=ax)
        st.pyplot(fig)

# Fonction pour afficher les prédictions
def afficher_predictions():
    st.title("Prédiction d'Attrition des Employés")
    st.write("Veuillez charger votre fichier CSV :")

    uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.write(df.head())

        if st.button("Prédire"):
            y_pred = predict(df)
            if y_pred[0] == 1:
                st.error("L'employé va probablement quitter le travail.")
            else:
                st.success("L'employé ne va probablement pas quitter le travail.")

# Sélection pour le tableau de bord ou les prédictions
selection = st.sidebar.selectbox("Choisissez une option :", ("Dashboard", "Prédictions"))

if selection == "Dashboard":
    afficher_dashboard()
elif selection == "Prédictions":
    afficher_predictions()
