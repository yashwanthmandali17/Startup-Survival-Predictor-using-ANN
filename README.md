# 🚀 Startup Survival Predictor — ANN + Supabase + Streamlit

Predicts whether a startup will **operate**, be **acquired**, or **close**
using an Artificial Neural Network trained on 2,000 Crunchbase-style companies.

---

## 📁 Project Structure

```
startup_ann_project/
├── app.py                        # Streamlit application (4 pages)
├── requirements.txt              # Python dependencies
├── schema.sql                    # Supabase table setup SQL
├── data/
│   ├── startup_data.csv          # Auto-generated dataset (2000 rows)
│   ├── generate_dataset.py       # Dataset generator
│   └── eda_plots/                # 9 EDA + model plots (PNG)
├── models/
│   ├── train_ann.py              # ANN training script
│   └── ann_model.pkl             # Saved model artifacts
└── utils/
    ├── eda_preprocessing.py      # EDA + cleaning pipeline
    ├── predictor.py              # Inference + insights engine
    └── supabase_helper.py        # Supabase REST integration
```

---

## ⚡ Quick Start (Step-by-Step)

### Step 1 — Clone / Download the project

```bash
cd startup_ann_project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up Supabase (ONE TIME)

1. Go to [https://supabase.com](https://supabase.com) and open your project
2. Navigate to **SQL Editor → New Query**
3. Paste and run the contents of `schema.sql`
4. This creates the `predictions` table

### Step 5 — Generate the dataset

```bash
python data/generate_dataset.py
```

Output: `data/startup_data.csv` (2000 rows, 18 columns)

### Step 6 — Train the ANN model

```bash
python models/train_ann.py
```

This will:
- Run full EDA and save 9 plots to `data/eda_plots/`
- Train an MLP Classifier (128→64→32→Softmax)
- Print accuracy, ROC-AUC, and classification report
- Save `models/ann_model.pkl`

Expected output:
```
TEST ACCURACY : ~82%
ROC-AUC (OVR) : ~0.61
5-Fold CV     : ~82% ± 0.8%
```

### Step 7 — Run the Streamlit app

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🏠 Home & Predict | Enter company details → ANN predicts status + insights + recommendations |
| 📊 EDA & Analytics | View all 9 EDA/model plots + dataset stats + feature correlations |
| 📜 Prediction History | View all saved predictions from Supabase database |
| ℹ️ About & Schema | Architecture, features, SQL schema, project structure |

---

## 🧠 ANN Architecture

```
Input (15 features)
    ↓
Dense Layer 1: 128 neurons, ReLU
    ↓
Dense Layer 2: 64 neurons, ReLU
    ↓
Dense Layer 3: 32 neurons, ReLU
    ↓
Output: 3 neurons, Softmax
   → operating / acquired / closed
```

- **Optimizer:** Adam (adaptive learning rate)
- **Regularization:** L2 alpha=0.001 + Early Stopping
- **Batch Size:** 64
- **Max Epochs:** 500

---

## 📊 Dataset Features (15 inputs)

| Feature | Type | Description |
|---------|------|-------------|
| company_age_years | Numeric | Derived from founded year |
| category | Categorical | 15 industry sectors |
| country | Categorical | 15 countries |
| funding_round | Categorical | Seed / Series A–D+ / IPO |
| funding_total_usd_M | Numeric | Total funding in USD million |
| num_funding_rounds | Numeric | Number of rounds completed |
| num_employees | Numeric | Team headcount |
| num_investors | Numeric | Total investors |
| num_milestones | Numeric | Operational milestones |
| num_relationships | Numeric | Strategic partnerships |
| top_tier_investor | Binary | Tier-1 VC backing |
| revenue_proxy | Ordinal 0–5 | Revenue stage |
| market_size | Ordinal 1–3 | Small / Medium / Large |
| has_patent | Binary | IP / patent held |
| founder_experience | Ordinal 1–5 | Founder experience score |

---

## 🗄️ Supabase Configuration

```python
SUPABASE_URL     = "https://ysousrfyfkpldtfvtihx.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

The app uses **pure stdlib** (`urllib.request`) for Supabase REST API calls —
no `supabase-py` dependency required.

---

## 🚀 Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add secrets if needed

---

## 📋 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ann_model.pkl not found` | Run `python models/train_ann.py` first |
| `startup_data.csv not found` | Run `python data/generate_dataset.py` first |
| Supabase 404/406 error | Run `schema.sql` in Supabase SQL Editor |
| Module not found | Activate venv and run `pip install -r requirements.txt` |
