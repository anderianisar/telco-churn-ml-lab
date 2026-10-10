
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Advisor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
SAMPLE_PATH = BASE_DIR / "sample_customers.csv"


# --------------------------------------------------
# 2. LOAD MODEL AND METADATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = load_model()
    meta = load_metadata()
except Exception as e:
    st.error(f"Could not load the model or metadata: {e}")
    st.stop()


FEATURE_COLS = meta["feature_columns"]
DEFAULT_THRESHOLD = float(meta.get("threshold", 0.30))


# --------------------------------------------------
# 3. RAW INPUT COLUMNS AND SERVING ENCODER
# --------------------------------------------------
RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges"
]


def prepare_input(raw, columns=FEATURE_COLS):
    """Encode raw customer rows into the model's expected columns."""
    missing = [col for col in RAW_COLS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = pd.get_dummies(raw[RAW_COLS])
    X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)


def predict_probabilities(raw):
    """Return predicted churn probabilities."""
    X = prepare_input(raw)
    return model.predict_proba(X)[:, 1]


def risk_label(probability):
    """Describe probability using fixed risk bands."""
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


# --------------------------------------------------
# 4. HEADER AND SIDEBAR
# --------------------------------------------------
st.title("📊 Churn Risk Advisor")

st.write(
    "Estimate telecom customer churn risk, explore customer "
    "scenarios, and score multiple customers from a CSV file."
)

st.info(
    "This tool supports decision-making. Predictions are estimates, "
    "not guarantees of customer behavior."
)

threshold = st.sidebar.slider(
    "Retention alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Customers at or above this predicted probability "
         "will be flagged for retention attention."
)

st.sidebar.header("Model Card")
st.sidebar.write("Model:", meta.get("model_name", "Unknown"))
st.sidebar.write("Version:", meta.get("version", "Unknown"))
st.sidebar.write("Features:", len(FEATURE_COLS))
st.sidebar.write("Cross-validation ROC-AUC:", meta.get("cv_auc", "N/A"))
st.sidebar.write("CV AUC standard deviation:", meta.get("cv_auc_std", "N/A"))
st.sidebar.write("Test ROC-AUC:", meta.get("test_auc", "N/A"))
st.sidebar.caption(
    "Low: probability below 30%; Medium: 30% to below 60%; "
    "High: 60% or above. The retention alert uses your selected threshold."
)


# --------------------------------------------------
# 5. TABS
# --------------------------------------------------
tab_single, tab_batch, tab_about = st.tabs(
    ["👤 Single Customer", "📁 Batch Predictions", "ℹ️ About the Model"]
)


# --------------------------------------------------
# 6. SINGLE CUSTOMER PREDICTION
# --------------------------------------------------
with tab_single:
    st.subheader("Customer Details")
    st.write("Enter the customer's information and click Predict.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen?", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing?", ["Yes", "No"])

        with col3:
            online_security = st.selectbox(
                "Online security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["Yes", "No", "No internet service"]
            )

        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

        st.caption(
            "Total charges are estimated as tenure × monthly charges. "
            "Actual historical billing totals may differ."
        )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        total_charges = round(tenure * monthly_charges, 2)

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:
            probability = float(predict_probabilities(customer)[0])
            flagged = probability >= threshold

            st.divider()
            st.subheader("Prediction Result")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Predicted churn probability",
                f"{probability:.1%}"
            )
            metric2.metric(
                "Risk category",
                risk_label(probability)
            )
            metric3.metric(
                "Retention alert",
                "Flagged" if flagged else "Not flagged"
            )

            st.progress(float(np.clip(probability, 0, 1)))

            if flagged:
                st.warning(
                    "This customer meets the selected retention-alert "
                    "threshold. Consider reviewing appropriate retention options."
                )
            else:
                st.success(
                    "This customer is below the selected retention-alert threshold."
                )

            st.caption(
                "The result is a model estimate, not a certainty or a causal explanation."
            )

            # ------------------------------------------
            # 7. WHAT-IF ANALYSIS
            # ------------------------------------------
            st.divider()
            st.subheader("What-If Analysis")
            st.write(
                "Compare predictions after changing one customer attribute "
                "at a time. Other details remain unchanged."
            )

            scenarios = [
                ("Current contract", "Contract", contract),
                ("One-year contract", "Contract", "One year"),
                ("Two-year contract", "Contract", "Two year"),
                (
                    "Automatic credit card",
                    "PaymentMethod",
                    "Credit card (automatic)"
                )
            ]

            scenario_rows = []
            seen = set()

            for label, feature, value in scenarios:
                key = (feature, value)
                if key in seen:
                    continue
                seen.add(key)

                alternative = customer.copy()
                alternative.loc[0, feature] = value
                alt_probability = float(
                    predict_probabilities(alternative)[0]
                )

                scenario_rows.append({
                    "Scenario": label,
                    "Predicted churn": round(alt_probability, 4),
                    "Change vs. current": round(
                        alt_probability - probability, 4
                    )
                })

            scenario_table = pd.DataFrame(scenario_rows)
            st.dataframe(
                scenario_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "What-if differences are model-based associations, not "
                "proof that changing a contract or payment method will cause "
                "churn to decrease."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# 8. BATCH PREDICTIONS
# --------------------------------------------------
with tab_batch:
    st.subheader("Score Multiple Customers")
    st.write(
        "Upload a CSV containing the original customer columns. "
        "The sample file can be downloaded below."
    )

    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH, "rb") as f:
            st.download_button(
                "Download 50-customer sample CSV",
                data=f.read(),
                file_name="sample_customers.csv",
                mime="text/csv"
            )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)

            missing = [
                col for col in RAW_COLS
                if col not in batch_raw.columns
            ]

            if missing:
                st.error(
                    "Your CSV is missing these required columns: "
                    + ", ".join(missing)
                )
            elif batch_raw.empty:
                st.warning("The uploaded CSV contains no customer rows.")
            else:
                st.write(f"Rows uploaded: {len(batch_raw)}")

                if st.button("Score Customers", type="primary"):
                    batch_probabilities = predict_probabilities(batch_raw)

                    results = batch_raw.copy()
                    results["ChurnProbability"] = batch_probabilities
                    results["RiskCategory"] = [
                        risk_label(float(p))
                        for p in batch_probabilities
                    ]
                    results["RetentionAlert"] = [
                        "Flagged" if p >= threshold else "Not flagged"
                        for p in batch_probabilities
                    ]

                    st.subheader("Batch Results")

                    total_flagged = int(
                        (batch_probabilities >= threshold).sum()
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Customers scored", len(results))
                    m2.metric("Retention alerts", total_flagged)
                    m3.metric(
                        "Average predicted churn",
                        f"{np.mean(batch_probabilities):.1%}"
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_bytes = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download prediction results",
                        data=csv_bytes,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Could not process this CSV: {e}")


# --------------------------------------------------
# 9. MODEL INFORMATION AND LIMITATIONS
# --------------------------------------------------
with tab_about:
    st.subheader("Model Card")

    st.markdown("""
    **Intended use:** Help a telecom retention team prioritize customers
    for further review based on predicted churn risk.

    **Model:** XGBoost classifier.

    **Dataset:** IBM Telco Customer Churn, 7,043 customers.

    **Evaluation:** Five-fold cross-validation and a held-out test set.

    **Limitations:**
    - The model reflects patterns in its historical training data.
    - The dataset may not represent customers in Pakistan or other markets.
    - Customer plans, prices, and behavior can change over time.
    - Predicted associations are not proof of causation.
    - Predictions should support human review, not replace it.
    """)

    st.write("Cross-validation ROC-AUC:", meta.get("cv_auc", "Not provided"))
    st.write("CV standard deviation:", meta.get("cv_auc_std", "Not provided"))
    st.write("Test ROC-AUC:", meta.get("test_auc", "Not provided"))
    st.write("Decision threshold:", threshold)
    st.write("Model version:", meta.get("version", "1.0"))

    st.caption(
        "Version 1.0 — Model performance on new customers may differ "
        "from the reported evaluation results."
    )
