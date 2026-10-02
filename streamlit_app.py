import streamlit as st
import pandas as pd
import joblib
from io import BytesIO



# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Adult Census Income Predictor")
st.caption(
    "Random Forest Classification | UCI Adult Census Income Dataset"
)


# --------------------------------------------------
# SNOWFLAKE SESSION
# --------------------------------------------------

conn = st.connection("snowflake")
session = conn.session()
# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = (
        "@ML_PROJECT.INCOME_PREDICTION."
        "MODEL_STAGE/income_model.pkl"
    )

    with session.file.get_stream(model_path) as stream:
        model_bytes = stream.read()

    return joblib.load(BytesIO(model_bytes))


try:
    model = load_model()
    st.success("✅ Trained Random Forest model loaded")

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.header("👤 Enter Person Details")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        17,
        90,
        30
    )

    workclass = st.selectbox(
        "Work Class",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool"
        ]
    )

    education_num = st.slider(
        "Education Level",
        1,
        16,
        10
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ]
    )

    relationship = st.selectbox(
        "Relationship",
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried"
        ]
    )


with col2:

    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces"
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "White",
            "Black",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other"
        ]
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    hours_per_week = st.slider(
        "Hours per Week",
        1,
        100,
        40
    )

    capital_gain = st.number_input(
        "Capital Gain",
        0,
        100000,
        0,
        step=500
    )

    capital_loss = st.number_input(
        "Capital Loss",
        0,
        5000,
        0,
        step=100
    )

    native_country = st.selectbox(
        "Native Country",
        [
            "United-States",
            "Mexico",
            "Philippines",
            "Germany",
            "Canada",
            "India",
            "England",
            "Cuba",
            "Jamaica",
            "Other"
        ]
    )


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Income",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "AGE": [age],

        "WORKCLASS": [workclass],

        "FNLWGT": [150000],

        "EDUCATION": [education],

        "EDUCATION_NUM": [education_num],

        "MARITAL_STATUS": [marital_status],

        "OCCUPATION": [occupation],

        "RELATIONSHIP": [relationship],

        "RACE": [race],

        "SEX": [sex],

        "CAPITAL_GAIN": [capital_gain],

        "CAPITAL_LOSS": [capital_loss],

        "HOURS_PER_WEEK": [hours_per_week],

        "NATIVE_COUNTRY": [native_country]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability = probabilities[
        list(classes).index(prediction)
    ]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.header("🎯 Prediction Result")

    if prediction == ">50K":

        st.success(
            "💰 Predicted Income: MORE THAN $50K"
        )

    else:

        st.info(
            "💵 Predicted Income: $50K OR LESS"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction Confidence",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Model",
            "Random Forest"
        )

    st.progress(float(probability))

    with st.expander("📋 View Input Details"):

        st.dataframe(
            input_data.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "End-to-End ML Workflow in Snowflake | "
    "UCI Adult Census Income Dataset"
)