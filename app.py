import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, mean_squared_error, r2_score


# Building Pipeline
def build_pipeline(model, X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# Model Training 

def train_and_evaluate(model_class, X_train, X_test, y_train, y_test, is_classification=True):
    pipeline = build_pipeline(model_class, X_train)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader("Evaluation Results")
    
    if is_classification:

        st.write(f"**Accuracy Score:** {accuracy_score(y_test, y_pred):.3f}")

        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.text("Confusion Matrix:")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    else:
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.3f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")



# Streamlit UI 
st.set_page_config(page_title='ML Explorer', page_icon="ml-explorer-icon.png", layout="wide")
st.title("ML Explorer")
st.sidebar.header("Model & Preprocessing Controls")

problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])
if problem_type == "Classification":
    model_selection = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"])
else:
    model_selection = st.sidebar.selectbox("Select Model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])

remove_outliers = st.sidebar.checkbox("Remove Outliers (Z-Score)", value=False)

st.write("""
### Explore different Supervised machine learning models and see how they perform on your data!
Upload your dataset or use a sample one, choose a model, and get instant evaluation metrics.
""")

st.subheader("Upload Your Dataset")

dataset_option = st.radio(
    "Choose Dataset Source",
    ("Upload CSV", "Use Sample Iris Dataset")
)

df = None
if dataset_option == "Use Sample Iris Dataset":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = df['target'].astype(str)
    st.success("Sample Iris dataset loaded!")
elif dataset_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully!")


# Data Processing
if df is not None:
    st.write("First 5 rows of Dataset")
    st.write(df.head())

    st.subheader("Data Types and Missing Values")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue(), language='text')

    st.subheader("Statistical Facts")
    st.write(df.describe())

    st.sidebar.subheader("Variable Selection")
    target_variable = st.sidebar.selectbox("Select Target Variable (y)", options=df.columns.tolist())
    feature_variables = st.sidebar.multiselect( "Select Feature Variables (X)", options=df.columns.tolist(), default=df.columns.tolist()) # By default all columns are selected 

    if st.sidebar.button("Train Model"):
        st.write("Training the model...")

        X = df[feature_variables]
        y = df[target_variable]

        if problem_type == "Classification":
            y = y.astype(str)

        # Outlier Detection 
        if remove_outliers:
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            z_scores = np.abs((X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std())
            mask = (z_scores < 3).all(axis=1)
            X = X[mask]
            y = y[mask]
            st.info(f"Outliers removed using Z-score. Remaining samples: {X.shape[0]}")

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.write("---")
        st.header("Model Training Summary")
        st.write(f"**Model Selected:** {model_selection}")
        st.write(f"**Target Variable:** {target_variable}")
        st.info(f"""
            **Training Set:** {X_train.shape[0]} rows  
            **Testing Set:** {X_test.shape[0]} rows  
            - Missing values are imputed with **mean** (numeric cols) and **mode** (categorical cols)  
            - Values are scaled using **Standard Scaler**  
            - Outliers are handled (*if selected*)  
            - Categorical features are encoded with **One Hot Encoding**
            """)


        # Model Selection  
        if problem_type == "Classification":
            if model_selection == "Logistic Regression":
                train_and_evaluate(LogisticRegression(max_iter=1000), X_train, X_test, y_train, y_test)
            elif model_selection == "Decision Tree":
                train_and_evaluate(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)
            elif model_selection == "Random Forest":
                train_and_evaluate(RandomForestClassifier(), X_train, X_test, y_train, y_test)
            elif model_selection == "SVM":
                train_and_evaluate(SVC(), X_train, X_test, y_train, y_test)
        else:
            if model_selection == "Linear Regression":
                train_and_evaluate(LinearRegression(), X_train, X_test, y_train, y_test, is_classification=False)
            elif model_selection == "Decision Tree Regressor":
                train_and_evaluate(DecisionTreeRegressor(), X_train, X_test, y_train, y_test, is_classification=False)
            elif model_selection == "Random Forest Regressor":
                train_and_evaluate(RandomForestRegressor(), X_train, X_test, y_train, y_test, is_classification=False)
