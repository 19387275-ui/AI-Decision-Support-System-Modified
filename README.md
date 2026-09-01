# AI Decision Support System using Case-Based Reasoning

## System Overview
This project implements an AI-powered decision support system designed to help users to come up with informed decisions with the help of argumentative intelligence solutions like Case-Based Reasoning (CBR) or Bayesian Belief Networks (BBN).It leverages Python programming libraries like numpy,pandas and scikitlearn to map the request of users with previous use cases which have been stored in memory. The system is modular, allowing for easy expansion and integration with existing workflows.

## How to Run the System
To get this system up and running, please follow these steps:

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### 2. Clone the Repository
```bash
git clone [Your Repository URL]
cd Dissertation-AI-Decision-Support-System-Improved
```

### 3. Install Dependencies
Navigate to the project root directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Data Setup
The dataset was gathered from Kaggle, an open source platform, an it contained information about loan approvals for a variety of individuals.
The dataset was stored in the "data" folder and could be accessed using the path : "C:/Dissertation-AI-Decision-Support-System/Dissertation-AI-Decision-Support-System-Improved/dataset"

### 5. Run the Application
To start the main application, execute the `app.py` file located in the `src` directory:
```bash
python src/app.py
```
[Add any further instructions if the app runs as a web server, requires specific arguments, etc.]

## Working of `app.py`
The `app.py` file serves as the main entry point for the application. It typically handles [mention main responsibilities, e.g., setting up the web server, defining API endpoints, initializing ML models, processing user input]. Below is the code from `app.py`, which provides a detailed look into its structure and functionality:

```python
"""Streamlit front end for the CBR loan decision-support prototype."""

from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

from src.cbr_model import get_recommendation
from src.config import RECOMMENDATIONS_DIR

st.set_page_config(page_title="AI Loan Decision Support System", page_icon="🏦", layout="wide")
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = RECOMMENDATIONS_DIR / "recommendation_history.csv"

st.title("AI-Based Loan Approval Decision Support System")
st.caption("Case-Based Reasoning (CBR) recommendation engine")

with st.sidebar:
    st.header("System Performance")
    st.info("Evaluation metrics are reported in outputs/evaluation/evaluation_results.txt. They should not be interpreted as a guarantee of real-world loan approval.")

st.subheader("Applicant Information")
col1, col2 = st.columns(2)
with col1:
    no_of_dependents = st.number_input("Number of Dependents", 0, 5, 2)
    education_label = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employed_label = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income", 200_000, 10_000_000, 5_000_000, step=100_000)
    loan_amount = st.number_input("Loan Amount Requested", 300_000, 40_000_000, 15_000_000, step=100_000)
with col2:
    loan_term = st.number_input("Loan Term (years)", 2, 20, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 750)
    residential_assets_value = st.number_input("Residential Assets Value", 0, 30_000_000, 5_000_000, step=100_000)
    commercial_assets_value = st.number_input("Commercial Assets Value", 0, 20_000_000, 3_000_000, step=100_000)
    luxury_assets_value = st.number_input("Luxury Assets Value", 0, 40_000_000, 10_000_000, step=100_000)
bank_asset_value = st.number_input("Bank Asset Value", 0, 15_000_000, 4_000_000, step=100_000)

if st.button("Get Recommendation", type="primary"):
    applicant = {
        "no_of_dependents": no_of_dependents,
        "education": int(education_label == "Graduate"),
        "self_employed": int(employed_label == "Yes"),
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    result = get_recommendation(applicant)

    record = {**applicant, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "recommendation": result["recommendation"], "confidence": round(result["confidence"], 2)}
    pd.DataFrame([record]).to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)

    st.subheader("Decision Recommendation")
    if result["recommendation"] == "APPROVED":
        st.success("Loan Recommended for Approval")
    else:
        st.error("Loan Recommended for Rejection")

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{result['confidence']:.2f}%")
    m2.metric("Approved Neighbours", result["approved_neighbours"])
    m3.metric("Rejected Neighbours", result["rejected_neighbours"])

    st.subheader("Top 5 Similar Historical Cases")
    display = result["similar_cases"][["case_id", "loan_status", "similarity_percent"]].copy()
    display["loan_status"] = display["loan_status"].map({1: "Approved", 0: "Rejected"})
    display = display.rename(columns={"case_id": "Case ID", "loan_status": "Decision", "similarity_percent": "Similarity (%)"})
    display["Similarity (%)"] = display["Similarity (%)"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("This prototype provides a recommendation based on historical similarity. It is not a substitute for regulated lending decisions or human review.")

if HISTORY_FILE.exists():
    with st.sidebar:
        st.subheader("Recent Recommendations")
        st.dataframe(pd.read_csv(HISTORY_FILE).tail(10), use_container_width=True, hide_index=True)

```

## Project Structure
- `src/`: Contains all source code, including `app.py`.
- `dataset/`: Stores raw or processed data.
- `outputs/`: Where generated reports, models, or visualizations are saved.
- `tests/`: Unit and integration tests for the project.
- `requirements.txt`: Lists all Python dependencies.
- `README.md`: This file, providing an overview and instructions.
