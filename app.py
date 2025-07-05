import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title='ML Explorer', page_icon="ml-explorer-icon.png", layout="wide")
st.title("🔍 ML Explorer")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("📋 Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)
    st.write(df.describe())
    st.write("🔍 Missing values:")
    st.write(df.isnull().sum())

    st.subheader("🧹 Impute Missing Values")
    impute_option = st.selectbox("Choose imputation method", ["None", "Mean", "Median", "Mode"])
    if impute_option != "None":
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if impute_option == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif impute_option == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif impute_option == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)

    st.subheader("📦 Outlier Detection")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_outlier = st.selectbox("Select column", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=selected_outlier)
    st.pyplot(fig)

    Q1 = df[selected_outlier].quantile(0.25)
    Q3 = df[selected_outlier].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[selected_outlier] < Q1 - 1.5*IQR) | (df[selected_outlier] > Q3 + 1.5*IQR)]
    st.write("Detected Outliers:")
    st.write(outliers)

    st.subheader("📈 Custom Visualizations")

    plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Scatter Plot"])

    if plot_type == "Histogram":
        hist_col = st.selectbox("Select numeric column for Histogram", df.select_dtypes(include=['int64', 'float64']).columns)
        fig, ax = plt.subplots()
        sns.histplot(df[hist_col], kde=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_title(f"Histogram of {hist_col}")
        st.pyplot(fig)
    
    elif plot_type == "Scatter Plot":
        col1 = st.selectbox("X-axis", df.columns, key="scatter_x")
        col2 = st.selectbox("Y-axis", df.columns, key="scatter_y")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color="orange")
        ax.set_title(f"Scatter Plot: {col1} vs {col2}")
        st.pyplot(fig)

    
    st.subheader("🎯 Select Target Column")
    target_column = st.selectbox("Choose target", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Fill missing values
    X = X.fillna(0)
    y = y.fillna(0)


    # Encode non-numeric columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])


    test_size = st.slider("🧪 Test size", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.subheader("🤖 Choose Model")
    model_choice = st.selectbox("Select Model", [
        "Logistic Regression", "Random Forest Classifier",
            "Decision Tree Classifier", "K-Nearest Neighbors",
            "Support Vector Machine", "Naive Bayes",  "Linear Regression", "Random Forest Regressor",
            "Decision Tree Regressor"
    ])

    if st.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest Classifier":
            model = RandomForestClassifier()
        elif model_choice == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif model_choice == "Support Vector Machine":
            model = SVC()
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor()
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("✅ Model Results")
        if "Regressor" in model_choice or model_choice == "Linear Regression":
            # Regression metrics
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        else:
            # Classification metrics
            st.subheader("✅ Classification Results")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            st.pyplot(fig)

 
