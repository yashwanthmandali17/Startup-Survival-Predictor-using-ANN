"""
app.py  —  Startup Survival Predictor
Streamlit multi-page application using ANN + Supabase
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os, sys, warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from utils.predictor import (
    load_artifacts, predict,
    CATEGORIES, COUNTRIES, FUNDING_ROUNDS,
    STATUS_EMOJI, STATUS_LABEL
)
from utils.supabase_helper import (
    insert_prediction, fetch_predictions, test_connection, SCHEMA_SQL
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Survival Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    text-align: center;
  }
  .main-header h1 { color: #e94560; margin: 0; font-size: 2.4rem; }
  .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1.1rem; }

  .status-card {
    padding: 1.5rem; border-radius: 12px; text-align: center;
    font-size: 1.6rem; font-weight: 700; margin: 1rem 0;
  }
  .operating { background: linear-gradient(135deg,#1a472a,#2d6a4f); color: #b7e4c7; }
  .acquired  { background: linear-gradient(135deg,#1e3a5f,#2563eb); color: #bfdbfe; }
  .closed    { background: linear-gradient(135deg,#7f1d1d,#b91c1c); color: #fecaca; }

  .insight-box {
    background: #1e293b; border-left: 4px solid #e94560;
    padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0;
    color: #e2e8f0; font-size: 0.92rem;
  }
  .rec-box {
    background: #0f2d40; border-left: 4px solid #22d3ee;
    padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0;
    color: #e2e8f0; font-size: 0.92rem;
  }
  .metric-card {
    background: #1e293b; padding: 1rem; border-radius: 10px;
    text-align: center;
  }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────
def ensure_model_exists():
    import sys
    _BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _BASE)
    model_path   = os.path.join(_BASE, "models", "ann_model.pkl")
    dataset_path = os.path.join(_BASE, "data",   "startup_data.csv")
    # Generate dataset if missing
    if not os.path.exists(dataset_path):
        with st.spinner("Generating dataset for first run..."):
            os.makedirs(os.path.join(_BASE, "data"), exist_ok=True)
            exec(open(os.path.join(_BASE, "data", "generate_dataset.py")).read())
    # Train model if missing
    if not os.path.exists(model_path):
        with st.spinner("Training ANN model for first run (~30 sec)..."):
            os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
            from models.train_ann import train_and_evaluate
            train_and_evaluate(dataset_path)

@st.cache_resource
def get_artifacts():
    ensure_model_exists()
    return load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Startup Predictor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home & Predict",
        "📊 EDA & Analytics",
        "📜 Prediction History",
        "ℹ️  About & Schema"
    ])
    st.markdown("---")
    st.markdown(
        "<small>Powered by **ANN (MLPClassifier)**  \n"
        "Architecture: 128 → 64 → 32 → Softmax  \n"
        "Dataset: 2000 Crunchbase-style startups</small>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
#  PAGE 1: HOME & PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.markdown("""
    <div class="main-header">
      <h1>🚀 Startup Survival Predictor</h1>
      <p>Enter your company details and let our ANN model predict its future status</p>
    </div>
    """, unsafe_allow_html=True)

    arts = get_artifacts()

    # ── Model metrics banner ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Model Accuracy",  f"{arts['accuracy']*100:.1f}%")
    c2.metric("📊 ROC-AUC",         f"{arts['roc_auc']:.3f}")
    c3.metric("🔁 CV Mean Acc",     f"{arts['cv_mean']*100:.1f}%")
    c4.metric("📐 CV Std Dev",      f"±{arts['cv_std']*100:.2f}%")

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────
    st.subheader("📋 Company Information")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏢 Company Profile**")
            company_name = st.text_input("Company Name", value="My Startup Co.")
            category     = st.selectbox("Industry Category", CATEGORIES, index=CATEGORIES.index("SaaS"))
            country      = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("India"))
            founded_year = st.slider("Founded Year", 2000, 2024, 2019)
            company_age  = 2024 - founded_year
            st.info(f"Company Age: **{company_age} years**")

        with col2:
            st.markdown("**💰 Funding Details**")
            funding_round     = st.selectbox("Current Funding Round", FUNDING_ROUNDS)
            funding_usd       = st.number_input("Total Funding (USD Million)", 0.0, 5000.0, 3.0, step=0.5)
            num_fund_rounds   = st.slider("Number of Funding Rounds", 1, 10, 2)
            num_investors     = st.slider("Number of Investors", 0, 30, 4)
            top_tier_investor = st.checkbox("Backed by Top-Tier Investor?", value=False)

            st.markdown("**📈 Business Metrics**")
            revenue_proxy = st.slider(
                "Revenue Stage (0=None → 5=Strong)", 0, 5, 2,
                help="0=No revenue, 1=Pre-revenue, 2=Early, 3=Growing, 4=Stable, 5=Strong"
            )

        with col3:
            st.markdown("**👥 Team & Operations**")
            num_employees      = st.number_input("Number of Employees", 1, 100000, 45, step=5)
            founder_experience = st.slider("Founder Experience (1–5)", 1, 5, 3)
            has_patent         = st.checkbox("Has Patent / IP?", value=False)

            st.markdown("**🌐 Market & Traction**")
            market_size = st.selectbox(
                "Market Size",
                options=[1, 2, 3],
                format_func=lambda x: {1:"Small/Niche", 2:"Medium", 3:"Large/Scalable"}[x]
            )
            num_milestones    = st.slider("Milestones Achieved", 0, 30, 7)
            num_relationships = st.slider("Strategic Partnerships / Relationships", 0, 20, 5)

        submitted = st.form_submit_button("🔮 Predict Company Status", use_container_width=True)

    # ── Prediction output ─────────────────────────────────────
    if submitted:
        user_input = {
            "company_age_years":   company_age,
            "category":            category,
            "country":             country,
            "funding_round":       funding_round,
            "funding_total_usd_M": funding_usd,
            "num_funding_rounds":  num_fund_rounds,
            "num_employees":       int(num_employees),
            "num_investors":       num_investors,
            "num_milestones":      num_milestones,
            "num_relationships":   num_relationships,
            "top_tier_investor":   int(top_tier_investor),
            "revenue_proxy":       revenue_proxy,
            "market_size":         market_size,
            "has_patent":          int(has_patent),
            "founder_experience":  founder_experience
        }

        with st.spinner("🧠 Running ANN inference..."):
            result = predict(user_input, arts)

        status = result["predicted_status"]

        # ── Result card ───────────────────────────────────────
        st.markdown(f"""
        <div class="status-card {status}">
          {result['status_emoji']} &nbsp; {company_name.upper()} &nbsp;→&nbsp; {result['status_label'].upper()}
          <br><small style="font-size:1rem;font-weight:400;">Confidence: {result['confidence']*100:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability gauge ─────────────────────────────────
        st.subheader("📊 Status Probabilities")
        prob_df = pd.DataFrame({
            "Status":      list(result["probabilities"].keys()),
            "Probability": [v * 100 for v in result["probabilities"].values()]
        }).sort_values("Probability", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.dataframe(
                prob_df.style
                    .format({"Probability": "{:.1f}%"})
                    .background_gradient(subset=["Probability"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True
            )
        with col_b:
            for _, row in prob_df.iterrows():
                st.metric(
                    label=f"{STATUS_EMOJI.get(row['Status'],'')} {row['Status'].capitalize()}",
                    value=f"{row['Probability']:.1f}%"
                )

        # ── Insights & Recommendations ────────────────────────
        col_ins, col_rec = st.columns(2)
        with col_ins:
            st.subheader("💡 Key Insights")
            for ins in result["key_insights"]:
                st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

        with col_rec:
            st.subheader("🎯 Strategic Recommendations")
            for rec in result["recommendations"]:
                st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)

        # ── Company summary table ─────────────────────────────
        st.subheader("📋 Company Summary")
        summary_data = {
            "Field": [
                "Company Name","Industry","Country","Founded Year","Age",
                "Funding Round","Total Funding","Investors","Employees",
                "Revenue Stage","Market Size","Milestones","Predicted Status","Confidence"
            ],
            "Value": [
                company_name, category, country, founded_year, f"{company_age} yrs",
                funding_round, f"${funding_usd:.1f}M", num_investors, int(num_employees),
                f"{revenue_proxy}/5", {1:"Small",2:"Medium",3:"Large"}[market_size],
                num_milestones,
                f"{result['status_emoji']} {status.capitalize()}",
                f"{result['confidence']*100:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # ── Save to Supabase ──────────────────────────────────
        with st.spinner("💾 Saving to Supabase..."):
            record = {
                "company_name":        company_name,
                "founded_year":        int(founded_year),
                "category":            category,
                "country":             country,
                "funding_round":       funding_round,
                "funding_usd_m":       float(funding_usd),
                "num_funding_rounds":  int(num_fund_rounds),
                "num_employees":       int(num_employees),
                "num_investors":       int(num_investors),
                "num_milestones":      int(num_milestones),
                "num_relationships":   int(num_relationships),
                "top_tier_investor":   bool(top_tier_investor),
                "revenue_proxy":       float(revenue_proxy),
                "market_size":         int(market_size),
                "has_patent":          bool(has_patent),
                "founder_experience":  int(founder_experience),
                "predicted_status":    status,
                "confidence":          round(result["confidence"], 4),
                "key_insights":        " | ".join(result["key_insights"]),
                "recommendations":     " | ".join(result["recommendations"])
            }
            db_result = insert_prediction(record)

        if "error" not in str(db_result).lower():
            st.success("✅ Prediction saved to Supabase database!")
        else:
            st.warning(f"⚠️ Could not save to Supabase — check table schema. Details: {db_result}")
            st.code(SCHEMA_SQL, language="sql")


# ═══════════════════════════════════════════════════════════════
#  PAGE 2: EDA & ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 EDA & Analytics":
    st.markdown("""
    <div class="main-header">
      <h1>📊 Exploratory Data Analysis</h1>
      <p>Dataset insights and visual analytics from 2,000 startups</p>
    </div>
    """, unsafe_allow_html=True)

    EDA_DIR = ('D:/try-2/data/eda_plots')
    CSV_PATH = ('D:/try-2/data/startup_data.csv')

    # Dataset snapshot
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        st.subheader("🗃️ Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{len(df):,}")
        c2.metric("Features",      f"{df.shape[1]-1}")
        c3.metric("Operating",     f"{(df['status']=='operating').sum()}")
        c4.metric("Missing Values",f"{df.isnull().sum().sum()}")

        with st.expander("📋 View Raw Dataset (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)

        with st.expander("📈 Descriptive Statistics"):
            st.dataframe(df.describe().T.round(2), use_container_width=True)

    # Plots
    plots = [
        ("01_status_distribution.png",  "Company Status Distribution"),
        ("02_correlation_heatmap.png",   "Feature Correlation Heatmap"),
        ("03_funding_by_status.png",     "Median Funding by Status"),
        ("04_category_status.png",       "Status Distribution by Industry"),
        ("05_feature_distributions.png", "Feature Distributions"),
        ("06_boxplots.png",              "Key Feature Boxplots by Status"),
        ("07_confusion_matrix.png",      "Model Confusion Matrix"),
        ("08_loss_curve.png",            "ANN Training Loss Curve"),
        ("09_feature_importance.png",    "Feature Importance (ANN Weights)"),
    ]

    tab_names = [name for _, name in plots]
    tabs = st.tabs(tab_names)
    for tab, (fname, title) in zip(tabs, plots):
        with tab:
            fpath = os.path.join(EDA_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=title)
            else:
                st.warning(f"Plot not found: {fname}. Run `python models/train_ann.py` first.")

    # Correlation insights
    if os.path.exists(CSV_PATH):
        st.subheader("🔍 Top Feature Correlations with Numeric Target")
        df2 = pd.read_csv(CSV_PATH)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df2["status_enc"] = le.fit_transform(df2["status"])
        num_cols = ["company_age_years","funding_total_usd_M","num_funding_rounds",
                    "num_employees","num_investors","num_milestones","num_relationships",
                    "top_tier_investor","revenue_proxy","market_size",
                    "has_patent","founder_experience","status_enc"]
        corr_with_target = df2[num_cols].corr()["status_enc"].drop("status_enc").sort_values(ascending=False)
        st.dataframe(
            corr_with_target.reset_index().rename(
                columns={"index":"Feature","status_enc":"Correlation with Status"}
            ).style.background_gradient(cmap="RdYlGn", subset=["Correlation with Status"]),
            use_container_width=True, hide_index=True
        )


# ═══════════════════════════════════════════════════════════════
#  PAGE 3: PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════
elif page == "📜 Prediction History":
    st.markdown("""
    <div class="main-header">
      <h1>📜 Prediction History</h1>
      <p>All company predictions stored in Supabase</p>
    </div>
    """, unsafe_allow_html=True)

    # Connection status
    connected = test_connection()
    if connected:
        st.success("🟢 Supabase Connected")
    else:
        st.error("🔴 Supabase Not Connected — check credentials or table schema")

    with st.spinner("Fetching predictions from Supabase..."):
        history = fetch_predictions(limit=100)

    if history:
        df_hist = pd.DataFrame(history)
        st.subheader(f"📋 {len(df_hist)} Predictions Found")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", len(df_hist))
        if "predicted_status" in df_hist.columns:
            c2.metric("🟢 Operating",  (df_hist["predicted_status"]=="operating").sum())
            c3.metric("🔵 Acquired",   (df_hist["predicted_status"]=="acquired").sum())
            c4.metric("🔴 Closed",     (df_hist["predicted_status"]=="closed").sum())

        # Display table
        display_cols = [
            "company_name","category","country","funding_usd_m",
            "num_employees","predicted_status","confidence","created_at"
        ]
        available = [c for c in display_cols if c in df_hist.columns]
        st.dataframe(
            df_hist[available].style.applymap(
                lambda v: "background-color:#1a472a;color:#b7e4c7" if v=="operating"
                else ("background-color:#1e3a5f;color:#bfdbfe" if v=="acquired"
                else ("background-color:#7f1d1d;color:#fecaca" if v=="closed" else "")),
                subset=[c for c in ["predicted_status"] if c in available]
            ),
            use_container_width=True, hide_index=True
        )

        # Expandable: insights + recommendations per record
        st.subheader("🔍 Detailed Records")
        for _, row in df_hist.iterrows():
            company = row.get("company_name", "Unknown")
            status  = row.get("predicted_status", "N/A")
            emoji   = STATUS_EMOJI.get(status, "⚪")
            with st.expander(f"{emoji} {company} — {status.capitalize()} ({row.get('confidence', 0)*100:.1f}%)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**💡 Key Insights**")
                    insights = str(row.get("key_insights","")).split(" | ")
                    for ins in insights:
                        if ins.strip():
                            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**🎯 Recommendations**")
                    recs = str(row.get("recommendations","")).split(" | ")
                    for rec in recs:
                        if rec.strip():
                            st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Go to 🏠 Home & Predict to make your first prediction!")
        st.markdown("**Note:** If the table doesn't exist in Supabase, run the SQL below first:")
        st.code(SCHEMA_SQL, language="sql")


# ═══════════════════════════════════════════════════════════════
#  PAGE 4: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️  About & Schema":
    st.markdown("""
    <div class="main-header">
      <h1>ℹ️ About This Project</h1>
      <p>ANN-powered company survival predictor with Supabase backend</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 ANN Architecture")
        st.markdown("""
        | Layer | Neurons | Activation |
        |-------|---------|------------|
        | Input | 15 | — |
        | Hidden 1 | 128 | ReLU |
        | Hidden 2 | 64 | ReLU |
        | Hidden 3 | 32 | ReLU |
        | Output | 3 | Softmax |

        - **Optimizer:** Adam (lr=0.001, adaptive)
        - **Regularization:** L2 α=0.001, Early Stopping
        - **Batch Size:** 64
        - **Max Epochs:** 500
        """)

        st.subheader("📦 Tech Stack")
        st.markdown("""
        - **ML:** scikit-learn MLPClassifier (ANN)
        - **Data:** pandas, numpy, seaborn, matplotlib
        - **Database:** Supabase (PostgreSQL)
        - **UI:** Streamlit
        - **Dataset:** 2000 Crunchbase-style startups
        """)

    with col2:
        st.subheader("🗂 Input Features (15)")
        feats = [
            ("company_age_years",      "Numeric",      "Age derived from founded year"),
            ("category",               "Categorical",  "Industry sector (15 categories)"),
            ("country",                "Categorical",  "Country of operation (15 countries)"),
            ("funding_round",          "Categorical",  "Seed / Series A–D+ / IPO"),
            ("funding_total_usd_M",    "Numeric",      "Total funding raised in USD million"),
            ("num_funding_rounds",     "Numeric",      "Number of completed rounds"),
            ("num_employees",          "Numeric",      "Headcount"),
            ("num_investors",          "Numeric",      "Total investor count"),
            ("num_milestones",         "Numeric",      "Operational milestones hit"),
            ("num_relationships",      "Numeric",      "Partnerships / key relationships"),
            ("top_tier_investor",      "Binary",       "Backed by Tier-1 VC?"),
            ("revenue_proxy",          "Ordinal 0–5",  "Revenue stage scale"),
            ("market_size",            "Ordinal 1–3",  "Small / Medium / Large market"),
            ("has_patent",             "Binary",       "IP/Patent held?"),
            ("founder_experience",     "Ordinal 1–5",  "Founder experience score"),
        ]
        st.dataframe(
            pd.DataFrame(feats, columns=["Feature","Type","Description"]),
            use_container_width=True, hide_index=True
        )