import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Project", layout="centered")

st.title("🤖 Machine Learning Project")
st.subheader("Select an ML Category")

choice = st.radio("Choose category", ["🧠 Supervised - Diabetes Prediction", "🌀 Unsupervised - Iris Clustering"])

# ---------------- SUPERVISED ----------------
if choice == "🧠 Supervised - Diabetes Prediction":
    st.header("🧠 Supervised Learning - Diabetes Prediction")

    @st.cache_data
    def load_diabetes_data():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        return pd.read_csv(url)

    data = load_diabetes_data()
    st.write("### 📊 Dataset Preview")
    st.write(data.head())

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### ✅ Accuracy:", accuracy_score(y_test, y_pred))
    st.write("### 📉 Confusion Matrix")
    st.dataframe(confusion_matrix(y_test, y_pred))

    st.write("### 🧾 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("---")
    st.write("### 🔍 Predict Diabetes")
    input_data = []
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        result = model.predict([input_data])[0]
        st.success("✅ Positive for Diabetes" if result == 1 else "❌ Negative for Diabetes")

# ---------------- UNSUPERVISED ----------------
elif choice == "🌀 Unsupervised - Iris Clustering":
    st.header("🌀 Unsupervised Learning - Iris Flower Clustering")

    @st.cache_data
    def load_iris_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        return df

    df = load_iris_data()
    st.write("### 🌸 Iris Dataset Preview")
    st.write(df.head())

    features = st.multiselect("Select features for clustering", df.columns.tolist(), default=df.columns.tolist()[:2])

    if len(features) >= 2:
        X = df[features]

        k = st.slider("Select number of clusters (K)", 2, 5, 3)
        kmeans = KMeans(n_clusters=k, random_state=0)
        df["Cluster"] = kmeans.fit_predict(X)

        st.write("### 🎯 Clustered Data Preview")
        st.write(df.head())

        fig, ax = plt.subplots()
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="viridis")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)
    else:
        st.warning("⚠️ Select at least two features to visualize clustering.")
