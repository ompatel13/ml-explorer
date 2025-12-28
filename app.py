import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------- CANDIDATE FUNCTION ---------------- #

def show_classification_results(y_test, y_pred, model_name="Model"):
    acc = accuracy_score(y_test, y_pred)
    st.success(f"{model_name} trained! Accuracy: {acc:.4f}")

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))



st.set_page_config(page_title="ML Explorer", layout="wide")

st.title("ML Explorer")

st.write("\n")
st.write("\n")
st.write("\n")

# ------------- DATA UPLOAD ------------- #
st.header("Upload The Classification Dataset")
uploaded_file = st.file_uploader(" ", type="csv",page_icon="ml-explorer-icon.png", accept_multiple_files=False)

st.write("\n")
st.write("\n")
st.write("\n")

df = None
X = None
y = None
X_train = X_test = y_train = y_test = None
target_col = None
categorical_transformer = None

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.info("Displaying First 10 Records From Dataset")
    st.write(df.head(10))

    st.write("\n")
    st.write("\n")
    st.write("\n")

    show_full_dataset = st.button("Show Full Dataset")
    
    if show_full_dataset:
        st.success("Displaying Full Dataset")
        st.write("\n")
        st.write("Shape : ", df.shape)

    st.write("\n")
    st.write("\n")
    st.write("\n")

# ------------- STATISTICAL CHARACTERISTIC OF DATASET ------------- #

    st.header("Statistical Characteristics Of Dataset")
    st.write(df.describe(include='all'))
    st.write("Data types : ")
    st.write(df.dtypes)


# ------------- CHECK CLASS IMBALANCE ------------- #
    st.header("Check Class Imbalance OR Not")

    cols_in_dataset = list(df.columns)
    target_col = st.selectbox(
        "Select Target Variable",
        cols_in_dataset,
        index=None,
        placeholder="Select Target Column"
    )

    if target_col is not None:

        st.bar_chart(df[target_col].value_counts())

        df = df.dropna(subset=[target_col])


# ------------- SPLIT DATASET WITH STRATIFIED SAMPLING TECHNIQUE ------------- #

        st.subheader("Train / Test Split Using Stratified Sampling")

        st.info(
            "Stratified sampling makes sure every class appears in the "
            "train and test sets in roughly the same proportions as the original data."
        )

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        st.write(
            "X_train shape:", X_train.shape,
            " | X_test shape:", X_test.shape,
            " | y_train shape:", y_train.shape,
            " | y_test shape:", y_test.shape
        )
        st.success("Dataset split using stratified sampling")

        st.write("\n")
        st.write("\n")
        st.write("\n")

# ------------- MODEL SELECTION & TRAINING ------------- #

    st.header("Select The Model You Want To Use")

    models_to_select = ["Logistic Regression", "Support Vector Machine", "Random Forest"]
    model_selected = st.selectbox(
    "Select The Classification Model",
    models_to_select,
    index=None,
    placeholder="Select Model"
    )   

    if model_selected is not None and X_train is not None and y_train is not None:

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

        # RANDOM FOREST
        if model_selected == "Random Forest":
            st.subheader("Random Forest Settings")

  
            rf_strategy = ["mean", "median", "most_frequent"]
            strategy_impute_rf = st.selectbox(
                "Select strategy to impute numeric values",
                rf_strategy
            )

            if len(categorical_cols) > 0:
                encoding_choice = st.selectbox(
                    "Select encoding technique for categorical features",
                    ["OneHotEncoder", "OrdinalEncoder"]
                )

                if encoding_choice == "OneHotEncoder":
                    cat_encoder = OneHotEncoder(handle_unknown="ignore")
                else:
                    cat_encoder = OrdinalEncoder()

        
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=strategy_impute_rf))
            ])

            # Preprocess categorical features (if any)
            if len(categorical_cols) > 0:
                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", cat_encoder)
                ])
            else:
                categorical_transformer = "drop"


            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols)
                ]
            )

            rf_clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    random_state=42
                ))
            ])

            if st.button("Train Random Forest"):
                rf_clf.fit(X_train, y_train)
                y_pred = rf_clf.predict(X_test)  

                show_classification_results(y_test, y_pred, "Random Forest")

        # LOGISTIC REGRESSION
        
        if model_selected == "Logistic Regression":
            st.subheader("Logistic Regression Settings")

            lr_strategy = ["mean", "median"]
            strategy_impute_lr = st.selectbox(
                "Select strategy to impute numeric values",
                lr_strategy
            )

            scaling_choice = st.selectbox(
                "Select scaling technique",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )

            if scaling_choice == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()  
            
            if len(categorical_cols) > 0:
                cat_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                )

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=strategy_impute_lr)),
                ("scaler", scaler)
            ])

            if len(categorical_cols) > 0:
                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", cat_encoder)
                ])
            else:
                categorical_transformer = "drop"

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols)
                ]
            )

            lr_clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                ))
            ])

            if st.button("Train Logistic Regression"):
                lr_clf.fit(X_train, y_train)
                y_pred = lr_clf.predict(X_test)

                show_classification_results(y_test, y_pred, "Logistic Regression")


        # Support Vector Machine 

        if model_selected == "Support Vector Machine":
            st.subheader("Support Vector Machine Settings")

            svm_strategy = ["mean", "median"]
            strategy_impute_svm = st.selectbox(
                "Select strategy to impute numeric values",
                svm_strategy
            )

            scaling_choice = st.selectbox(
                "Select scaling technique",
                ["StandardScaler", "RobustScaler"]
            )

            if scaling_choice == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()  

            if len(categorical_cols) > 0:
                cat_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                )
        
            kernel_choice = st.selectbox(
                "Select SVM kernel",
                ["linear", "rbf", "poly", "sigmoid"]
            )

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=strategy_impute_svm)),
                ("scaler", scaler)
            ])

            if len(categorical_cols) > 0:
                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", cat_encoder)
                ])
            else:
                categorical_transformer = "drop"

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols)
                ]
            )

            svm_clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", SVC(
                    kernel=kernel_choice,
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                ))
            ])

            if st.button("Train Support Vector Machine"):
                svm_clf.fit(X_train, y_train)
                y_pred = svm_clf.predict(X_test)

                show_classification_results(y_test, y_pred, "Support Vector Machine")

   


  
