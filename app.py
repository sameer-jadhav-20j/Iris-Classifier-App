import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
model = joblib.load("iris_model.pkl")

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

#  Page Setup 
st.set_page_config(
    page_title="Iris Flower Classifier",
    layout="centered"
)

st.title("Iris Flower Classifier")
st.write("Predict the species of an iris flower using a trained machine learning model")

#  Sidebar Inputs
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Input for prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)

# Prediction
prediction = model.predict(input_data)[0]
predicted_species = iris.target_names[prediction]

st.markdown("---")
st.subheader("Prediction")
st.write(f"The predicted species is:")
st.success(f"{predicted_species.title()}")

# Dataset Sample
with st.expander("Show Random Sample from Iris Dataset"):
    st.dataframe(df.sample(10), use_container_width=True)

# Visualization
with st.expander("View Feature Relationships (Pairplot)"):
    fig = sns.pairplot(df, hue="species", palette="Set2")
    st.pyplot(fig)

