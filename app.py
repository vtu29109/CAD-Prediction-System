import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "/content/cad_svm_pipeline.pkl"
DATA_PATH = "/content/heart disease CAD.xlsx"

st.set_page_config(
    page_title="CAD Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional white-and-teal styling
st.markdown("""
<style>
.stApp {
    background-color: #f5f9fa;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #087f8c, #13a6a0);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #efffff;
    font-size: 16px;
}
.section-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce9eb;
    margin-bottom: 12px;
}
.result-card {
    background: white;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #cfe2e5;
    text-align: center;
}
.small-note {
    color: #536b70;
    font-size: 13px;
}
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background-color: #087f8c;
    color: white;
    border-radius: 8px;
    min-height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


model = load_model()
data = load_data()

# Use the exact features and order from model training
feature_cols = list(model.feature_names_in_)

categorical_cols = [
    col for col in feature_cols
    if data[col].dtype == "object"
]
numerical_cols = [
    col for col in feature_cols
    if data[col].dtype != "object"
]

# Page heading
st.markdown("""
<div class="hero">
    <h1>🫀 CAD Prediction System</h1>
    <p>Machine-learning demonstration for Coronary Artery Disease
    classification using clinical dataset features.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only. This system is not a medical device "
    "and cannot diagnose or rule out CAD. Do not use its predictions "
    "for treatment or other medical decisions."
)

# Sidebar
with st.sidebar:
    st.title("🫀 CAD System")
    st.caption("Student project prototype")
    st.divider()

    st.subheader("About this project")
    st.write(
        "A Support Vector Machine (SVM) pipeline classifies the "
        "entered feature values into the classes learned during training."
    )

    st.divider()
    st.metric("Dataset records", len(data))
    st.metric("Model input features", len(feature_cols))
    st.metric("Numerical features", len(numerical_cols))
    st.metric("Categorical features", len(categorical_cols))

    st.caption(
        "The dataset contains 303 records. Dataset values and model "
        "predictions should not be interpreted as clinical guidance."
    )

st.subheader("Enter example details")

with st.expander("About the project and model", expanded=False):
    st.markdown("""
    ### Project objective
    Develop a machine-learning prototype that classifies example
    clinical dataset records into CAD or Normal categories.

    ### Model
    - **Algorithm:** Support Vector Machine (SVM)
    - **Dataset:** 303 records
    - **Input features:** 55
    - **Numerical features:** 34
    - **Categorical features:** 21

    ### Workflow
    1. Enter example feature values.
    2. The saved pipeline preprocesses the inputs.
    3. The SVM model predicts a class.
    4. The application displays the model output.

    ### Important limitation
    This is an educational prototype, not a validated clinical
    diagnostic system. Its predictions must not guide medical care.
    """)

st.write(
    "Complete the sections below. The initial values are taken from "
    "the dataset and can be changed for demonstration."
)

# Group the original model features into sections
sections = {
    "1. Demographic details": [
        "Age", "Sex", "Weight", "Length", "BMI"
    ],
    "2. Medical history and risk factors": [
        "DM", "HTN", "Current smokker", "Ex-smokker", "FH",
        "Obesity", "CRF", "CVA", "Airway disease",
        "Thyroid  disease", "CHF", "DLP"
    ],
    "3. Symptoms and clinical examination": [
        "BP", "PR", "Edema", "Weak peripheral plus",
        "Lung rales", "Systolic murmur", "Diastolic murmur",
        "Typical chest pain", "Dyspnea", "Function class",
        "Atypical", "Nonanginal", "Exertional CP", "Lowth ANG"
    ],
    "4. ECG-related features": [
        "Q wave", "St Elevation", "St Depression", "Tinversion",
        "LVH", "Poor R Progression", "BBB"
    ],
    "5. Laboratory measurements": [
        "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
        "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"
    ],
    "6. Cardiac assessment features": [
        "EF-TTE", "Region RWMA", "VHD"
    ]
}

# Confirm every input feature is represented exactly once
grouped_features = [
    feature
    for section_features in sections.values()
    for feature in section_features
]
if (
    len(grouped_features) != len(set(grouped_features))
    or set(grouped_features) != set(feature_cols)
):
    st.error(
        "The input sections do not match the saved model's features. "
        "Please check the section definitions."
    )
    st.stop()

numeric_values = {}
categorical_values = {}

with st.form("cad_form"):
    for section_name, section_features in sections.items():
        with st.expander(section_name, expanded=(section_name ==
                                                 "1. Demographic details")):
            columns = st.columns(2)

            for i, col in enumerate(section_features):
                with columns[i % 2]:
                    if col in numerical_cols:
                        values = pd.to_numeric(
                            data[col], errors="coerce"
                        ).dropna()

                        if values.empty:
                            minimum, maximum, default = 0.0, 100.0, 0.0
                        else:
                            minimum = float(values.min())
                            maximum = float(values.max())
                            default = float(values.median())

                        if minimum == maximum:
                            minimum -= 1.0
                            maximum += 1.0

                        is_integer = (
                            pd.api.types.is_integer_dtype(data[col])
                            or set(values.unique()).issubset({0, 1})
                        )

                        if is_integer:
                            minimum = float(int(minimum))
                            maximum = float(int(maximum))
                            default = float(round(default))
                            step = 1.0
                        else:
                            step = 0.1

                        numeric_values[col] = st.number_input(
                            col,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=step,
                            help=(
                                "Range is based on this dataset, "
                                "not a medical reference range."
                            )
                        )

                    else:
                        options = sorted(
                            data[col].dropna().astype(str).unique().tolist()
                        )

                        if not options:
                            options = ["Unknown"]

                        modes = data[col].dropna().astype(str).mode()
                        default = (
                            str(modes.iloc[0])
                            if not modes.empty else options[0]
                        )
                        default_index = (
                            options.index(default)
                            if default in options else 0
                        )

                        categorical_values[col] = st.selectbox(
                            col,
                            options=options,
                            index=default_index
                        )

    submitted = st.form_submit_button(
        "🧠 Predict CAD Class",
        use_container_width=True,
        type="primary"
    )

if submitted:
    input_values = {}
    input_values.update(numeric_values)
    input_values.update(categorical_values)

    # Preserve the exact column names and order used for training
    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_cols]],
        columns=feature_cols
    )

    try:
        prediction = model.predict(input_df)[0]
        prediction_text = str(prediction).strip().lower()

        is_cad = prediction_text in {"cad", "1", "1.0"}

        st.divider()
        st.subheader("Prediction result")

        if is_cad:
            st.markdown("""
            <div style="background-color:#fff1f0;
                        border:1px solid #e5a6a0;
                        border-radius:14px;padding:24px;text-align:center;">
                <h2 style="color:#a12622;">Model Output: CAD</h2>
                <p style="color:#63201c;">
                    The model classified the entered example into its CAD class.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#eaf7ef;
                        border:1px solid #9ac9a9;
                        border-radius:14px;padding:24px;text-align:center;">
                <h2 style="color:#216b3a;">Model Output: Normal</h2>
                <p style="color:#245536;">
                    The model classified the entered example into its Normal class.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "This is an experimental machine-learning classification, not a "
            "medical diagnosis. A Normal result does not rule out CAD, and a "
            "CAD result does not confirm it. Consult a qualified healthcare "
            "professional for medical concerns."
        )

        with st.expander("View submitted feature values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.write("Technical details:", str(error))

st.divider()
st.markdown(
    '<p class="small-note">B.Tech student project | '
    'SVM-based CAD classification | Research demonstration only</p>',
    unsafe_allow_html=True
)
