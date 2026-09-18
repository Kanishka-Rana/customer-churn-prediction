import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = None
features = None
result_df = None
missing = []
demo_file = None
df = None
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f8ff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.hero {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    padding: 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.20);
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
    font-weight: 750;
}

.hero p {
    font-size: 16px;
    opacity: 0.92;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 18px rgba(0,0,0,0.05);
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 750;
    margin-top: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-top: 28px;
    margin-bottom: 15px;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 15px;
    border-radius: 10px;
    color: #1e3a8a;
}

.success-box {
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    padding: 15px;
    border-radius: 10px;
    color: #065f46;
}

.warning-box {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 15px;
    border-radius: 10px;
    color: #92400e;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FIND DEMO DATASET
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Auto-search for csv anywhere inside project
possible_files = [
    BASE_DIR / "Data" / "Raw" / "customer_churn_dataset-1.csv",
    BASE_DIR / "Data" / "Raw" / "customer_churn_dataset.csv",
    BASE_DIR / "app" / "Data" / "Raw" / "customer_churn_dataset-1.csv",
    BASE_DIR / "app" / "Data" / "Raw" / "customer_churn_dataset.csv",
    BASE_DIR.parent / "app" / "Data" / "Raw" / "customer_churn_dataset-1.csv",
    Path.cwd() / "app" / "Data" / "Raw" / "customer_churn_dataset-1.csv",
    Path.cwd() / "Data" / "Raw" / "customer_churn_dataset-1.csv",
]

# Extra safety - kahi bhi churn csv mile to utha lo
if not any(f.exists() for f in possible_files):
    for p in BASE_DIR.rglob("*churn*.csv"):
        possible_files.append(p)
        break

demo_file = None
for file in possible_files:
    if file.exists():
        demo_file = file
        break


if demo_file is None:
    raw_folder = BASE_DIR / "Data" / "Raw"

    if raw_folder.exists():
        csv_files = list(raw_folder.glob("*.csv"))

        if len(csv_files) > 0:
            demo_file = csv_files[0]


# =========================================================
# FUNCTIONS
# =========================================================

def clean_data(df):
    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric-looking columns
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().mean() > 0.80:
                df[col] = converted

    return df


def find_target(df):
    possible_targets = [
        "Churn",
        "churn",
        "CHURN",
        "Churn_Flag",
        "ChurnFlag",
        "Exited",
        "Target"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    return None


def prepare_target(series):

    s = series.copy()

    # Already numeric 0/1
    if pd.api.types.is_numeric_dtype(s):

        unique = sorted(s.dropna().unique())

        if set(unique).issubset({0, 1}):
            return s.astype(int)

    # Text target
    s = s.astype(str).str.strip().str.lower()

    positive_values = [
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "exited"
    ]

    return s.isin(positive_values).astype(int)


def build_model(model_name):

    if model_name == "Logistic Regression":

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ])

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=8
        )

    else:

        return DecisionTreeClassifier(
            random_state=42,
            max_depth=7
        )


def train_model(df, target_col, model_name):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    features = [
        col for col in numeric_cols
        if col != target_col
    ]

    if len(features) == 0:
        return None, None, None, None

    model_df = df[features + [target_col]].dropna()

    if len(model_df) < 20:
        return None, None, None, None

    X = model_df[features]
    y = prepare_target(model_df[target_col])

    if y.nunique() < 2:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_model(model_name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    return model, features, metrics, X_test


def create_predictions(model, df, features):

    prediction_df = df.copy()

    usable_features = [
        col for col in features
        if col in prediction_df.columns
    ]

    if len(usable_features) != len(features):
        missing = [
            col for col in features
            if col not in prediction_df.columns
        ]

        return None, missing

    X = prediction_df[features].copy()

    for col in features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.fillna(X.median(numeric_only=True))

    prediction_df["Predicted Churn"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        prediction_df["Churn Probability"] = (
            probabilities * 100
        ).round(2)

    else:
        prediction_df["Churn Probability"] = np.nan

    prediction_df["Risk Level"] = np.where(
        prediction_df["Churn Probability"] >= 70,
        "High Risk",
        np.where(
            prediction_df["Churn Probability"] >= 40,
            "Medium Risk",
            "Low Risk"
        )
    )

    prediction_df["Prediction"] = np.where(
        prediction_df["Predicted Churn"] == 1,
        "Churn",
        "Stay"
    )

    return prediction_df, None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ Dashboard Settings")

st.sidebar.markdown("### 🤖 Prediction Model")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📂 Data Source")

data_source = st.sidebar.radio(
    "Choose dataset",
    [
        "Demo Dataset",
        "Upload CSV"
    ]
)

uploaded_file = None

if data_source == "Upload CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "💡 Upload a customer CSV to generate "
    "churn predictions, analyze customer behaviour "
    "and download the results."
)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>📊 Customer Churn Analytics</h1>

<p>
AI-powered customer retention dashboard for predicting churn,
understanding customer behaviour and identifying high-risk customers.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

df = None
data_name = ""

if data_source == "Demo Dataset":

    if demo_file is not None:

        try:
            df = pd.read_csv(demo_file)
            data_name = "Demo Dataset"

        except Exception as e:

            st.error(
                f"Unable to read demo dataset: {e}"
            )

    else:

        st.error(
            "Demo dataset not found. "
            "Please keep your CSV inside Data/Raw."
        )


else:

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)
            data_name = uploaded_file.name

        except Exception as e:

            st.error(
                f"Unable to read uploaded CSV: {e}"
            )

    else:

        st.markdown("""
        <div class="info-box">
        👆 Upload a customer CSV from the sidebar to start analysis.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# MAIN DASHBOARD
# =========================================================

if df is not None:

    df = clean_data(df)

    target_col = find_target(df)

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📁 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    info1, info2, info3, info4 = st.columns(4)

    with info1:
        st.metric(
            "Total Customers",
            f"{len(df):,}"
        )

    with info2:
        st.metric(
            "Features",
            len(df.columns)
        )

    with info3:
        st.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

    with info4:
        st.metric(
            "Dataset",
            data_name
        )

    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    with st.expander("🔍 View Dataset Preview", expanded=False):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.caption(
            f"Showing first 20 rows • {len(df)} total rows"
        )

    # -----------------------------------------------------
    # CHURN ANALYSIS
    # -----------------------------------------------------

    if target_col is not None:

        df["_ChurnNumeric"] = prepare_target(
            df[target_col]
        )

        total_customers = len(df)

        churned = int(
            df["_ChurnNumeric"].sum()
        )

        retained = total_customers - churned

        churn_rate = (
            churned / total_customers * 100
            if total_customers > 0
            else 0
        )

        st.markdown(
            '<div class="section-title">📈 Customer Churn Overview</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">TOTAL CUSTOMERS</div>
                <div class="metric-value">{total_customers:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">CHURNED CUSTOMERS</div>
                <div class="metric-value">{churned:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">RETAINED CUSTOMERS</div>
                <div class="metric-value">{retained:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">CHURN RATE</div>
                <div class="metric-value">{churn_rate:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # CHARTS
        # -------------------------------------------------

        chart1, chart2 = st.columns(2)

        with chart1:

            churn_counts = pd.DataFrame({
                "Status": [
                    "Retained",
                    "Churned"
                ],
                "Customers": [
                    retained,
                    churned
                ]
            })

            fig = px.pie(
                churn_counts,
                names="Status",
                values="Customers",
                hole=0.55,
                title="Customer Churn Distribution"
            )

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with chart2:

            fig2 = px.bar(
                churn_counts,
                x="Status",
                y="Customers",
                title="Churn vs Retention",
                text="Customers"
            )

            fig2.update_traces(
                textposition="outside"
            )

            fig2.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # -------------------------------------------------
        # FEATURE GRAPHS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Customer Behaviour Analysis</div>',
            unsafe_allow_html=True
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        numeric_columns = [
            col
            for col in numeric_columns
            if col != "_ChurnNumeric"
        ]

        if len(numeric_columns) > 0:

            selected_feature = st.selectbox(
                "Select a feature to analyse",
                numeric_columns
            )

            fig3 = px.histogram(
                df,
                x=selected_feature,
                color=target_col,
                nbins=30,
                title=f"{selected_feature} Distribution by Churn"
            )

            fig3.update_layout(
                height=450
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        # -------------------------------------------------
        # SPECIFIC BUSINESS GRAPHS
        # -------------------------------------------------

        preferred_features = [
            "Monthly_Charges",
            "Monthly Charges",
            "Support_Tickets",
            "Open_Tickets",
            "Avg_Resolution_Hours",
            "Payment_Delays",
            "Discount_Percent",
            "NPS_Score",
            "Days_Since_Last_Login"
        ]

        available_features = [
            col
            for col in preferred_features
            if col in df.columns
        ]

        if len(available_features) > 0:

            cols = st.columns(2)

            for i, feature in enumerate(
                available_features
            ):

                with cols[i % 2]:

                    fig4 = px.box(
                        df,
                        x=target_col,
                        y=feature,
                        color=target_col,
                        title=f"{feature} vs Churn"
                    )

                    fig4.update_layout(
                        height=380
                    )

                    st.plotly_chart(
                        fig4,
                        use_container_width=True
                    )

        # -------------------------------------------------
        # CORRELATION HEATMAP
        # -------------------------------------------------

        if len(numeric_columns) >= 2:

            st.markdown(
                '<div class="section-title">🔥 Feature Correlation</div>',
                unsafe_allow_html=True
            )

            corr = df[numeric_columns].corr()

            fig5 = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )

            fig5.update_layout(
                height=600
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

    else:

        st.warning(
            "⚠️ No Churn column was found in this dataset. "
            "Prediction can still work if the uploaded file "
            "contains the required customer features."
        )


    # =====================================================
    # MACHINE LEARNING SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">🤖 Churn Prediction Model</div>',
        unsafe_allow_html=True
    )

    if target_col is not None:

        model, features, metrics, X_test = train_model(
            df,
            target_col,
            model_name
        )

        if model is not None:

            st.markdown(
                f"""
                <div class="success-box">
                ✅ <b>{model_name}</b> trained successfully using
                the available customer data.
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------------------------------------------
            # MODEL METRICS
            # ---------------------------------------------

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy'] * 100:.1f}%"
                )

            with m2:
                st.metric(
                    "Precision",
                    f"{metrics['Precision'] * 100:.1f}%"
                )

            with m3:
                st.metric(
                    "Recall",
                    f"{metrics['Recall'] * 100:.1f}%"
                )

            with m4:
                st.metric(
                    "F1 Score",
                    f"{metrics['F1 Score'] * 100:.1f}%"
                )

            st.caption(
                "Model performance is calculated on a held-out test set."
            )

    
            # PREDICTIONS
result_df, missing = create_predictions(
                model,
                df,
                features
            )

if result_df is not None:

                st.markdown(
                    '<div class="section-title">🎯 Prediction Results</div>',
                    unsafe_allow_html=True
                )

                result_columns = [
                    col
                    for col in [
                        "Prediction",
                        "Churn Probability",
                        "Risk Level"
                    ]
                    if col in result_df.columns
                ]

                st.dataframe(
                    result_df[
                        result_columns
                    ].head(50),
                    use_container_width=True
                )

                # -----------------------------------------
                # RESULT SUMMARY
                # -----------------------------------------

                prediction_counts = (
                    result_df["Prediction"]
                    .value_counts()
                    .reset_index()
                )

                prediction_counts.columns = [
                    "Prediction",
                    "Customers"
                ]

                p1, p2 = st.columns(2)

                with p1:

                    fig6 = px.bar(
                        prediction_counts,
                        x="Prediction",
                        y="Customers",
                        color="Prediction",
                        title="Prediction Summary",
                        text="Customers"
                    )

                    fig6.update_layout(
                        height=400
                    )

                    st.plotly_chart(
                        fig6,
                        use_container_width=True
                    )

                with p2:

                    risk_counts = (
                        result_df["Risk Level"]
                        .value_counts()
                        .reset_index()
                    )

                    risk_counts.columns = [
                        "Risk Level",
                        "Customers"
                    ]

                    fig7 = px.pie(
                        risk_counts,
                        names="Risk Level",
                        values="Customers",
                        hole=0.5,
                        title="Customer Risk Levels"
                    )

                    fig7.update_layout(
                        height=400
                    )

                    st.plotly_chart(
                        fig7,
                        use_container_width=True
                    )

# -----------------------------------------
# DOWNLOAD RESULTS
# -----------------------------------------

st.markdown(
'<div class="section-title">⬇️ Download Results</div>',
unsafe_allow_html=True
)

## OUTER IF - did model train?
if result_df is not None and len(result_df) > 0:

    if not missing:   
        download_df = result_df.copy()

        if "_ChurnNumeric" in download_df.columns:
            download_df = download_df.drop(
                columns=["_ChurnNumeric"]
            )

        csv_data = download_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results CSV",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success(
            "Your prediction results are ready to download."
        )

    else:
        st.error(
            "Prediction could not be generated."
        )
        st.write(
            "Missing columns:",
            missing
        )

else:  
    st.error(
        "Unable to train the model. "
        "Make sure the dataset contains enough numeric "
        "customer features and both churn classes."
    )



# =====================================================
# DATASET DOWNLOAD
# =====================================================
st.markdown(
"""
<div class="section-title">📦 Dataset</div>
""",
unsafe_allow_html=True
)

dataset_csv = df.drop(
columns=["_ChurnNumeric"],
errors="ignore"
).to_csv(index=False).encode("utf-8")

st.download_button(
label="📥 Download Current Dataset",
data=dataset_csv,
file_name="customer_churn_dataset.csv",
mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:15px;">
    <b>Customer Churn Analytics</b><br>
    Machine Learning • Customer Behaviour • Predictive Analytics
    </div>
    """,
    unsafe_allow_html=True
    ) 