"""
predictor.py
Core prediction logic:
  - Load saved ANN model
  - Predict company status from user inputs
  - Generate key insights and strategic recommendations
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

import os
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE, "models", "ann_model.pkl")

# ── Category / Country / Round lists (must match training) ─────────
CATEGORIES = sorted([
    "Software", "Biotech", "E-Commerce", "FinTech", "EdTech",
    "HealthTech", "AI/ML", "CleanTech", "SaaS", "Hardware",
    "Media", "Gaming", "Logistics", "FoodTech", "AgriTech"
])

COUNTRIES = sorted([
    "USA", "India", "UK", "Germany", "China", "Canada",
    "France", "Israel", "Singapore", "Brazil", "Australia",
    "Japan", "South Korea", "Netherlands", "Sweden"
])

FUNDING_ROUNDS = ["Seed", "Series A", "Series B", "Series C", "Series D+", "IPO"]

STATUS_EMOJI = {
    "operating": "🟢",
    "acquired":  "🔵",
    "closed":    "🔴"
}

STATUS_LABEL = {
    "operating": "Likely to Withstand & Operate",
    "acquired":  "Likely to be Acquired",
    "closed":    "At Risk of Closing"
}


def load_artifacts() -> dict:
    return joblib.load(MODEL_PATH)


def encode_input(user_input: Dict[str, Any], artifacts: dict) -> np.ndarray:
    """
    Convert user inputs into the same format as training.
    user_input keys:
        company_age_years, category, country, funding_round,
        funding_total_usd_M, num_funding_rounds, num_employees,
        num_investors, num_milestones, num_relationships,
        top_tier_investor, revenue_proxy, market_size,
        has_patent, founder_experience
    """
    le_dict      = artifacts["label_encoders"]
    scaler       = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    row = {}
    for feat in feature_names:
        val = user_input.get(feat, 0)
        if feat in le_dict:
            try:
                val = le_dict[feat].transform([str(val)])[0]
            except ValueError:
                # unseen label → use most frequent class (index 0)
                val = 0
        row[feat] = val

    X = pd.DataFrame([row])[feature_names].values.astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled


def predict(user_input: Dict[str, Any], artifacts: dict) -> Dict[str, Any]:
    """Run prediction and return status, probabilities, insights, recommendations."""
    X = encode_input(user_input, artifacts)
    model     = artifacts["model"]
    target_le = artifacts["target_le"]
    classes   = artifacts["class_names"]

    probs      = model.predict_proba(X)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {c: float(p) for c, p in zip(classes, probs)}

    insights      = generate_insights(user_input, pred_class, prob_dict)
    recommendations = generate_recommendations(user_input, pred_class)

    return {
        "predicted_status": pred_class,
        "confidence":        confidence,
        "probabilities":     prob_dict,
        "status_label":      STATUS_LABEL[pred_class],
        "status_emoji":      STATUS_EMOJI[pred_class],
        "key_insights":      insights,
        "recommendations":   recommendations,
    }


def generate_insights(inp: dict, status: str, probs: dict) -> List[str]:
    """Generate dynamic key insights based on inputs and prediction."""
    insights = []
    funding  = inp.get("funding_total_usd_M", 0)
    rounds   = inp.get("num_funding_rounds", 1)
    emp      = inp.get("num_employees", 1)
    age      = inp.get("company_age_years", 1)
    invest   = inp.get("num_investors", 0)
    mile     = inp.get("num_milestones", 0)
    rev      = inp.get("revenue_proxy", 0)
    top_inv  = inp.get("top_tier_investor", 0)
    exp      = inp.get("founder_experience", 1)
    patent   = inp.get("has_patent", 0)
    mkt      = inp.get("market_size", 1)
    rel      = inp.get("num_relationships", 0)

    # Funding strength
    if funding > 50:
        insights.append(f"💰 Strong funding base of ${funding:.1f}M across {rounds} rounds signals high investor confidence.")
    elif funding < 2:
        insights.append(f"⚠️ Limited funding (${funding:.2f}M) is a key risk factor — restricted runway.")
    else:
        insights.append(f"💵 Moderate funding of ${funding:.1f}M provides a reasonable runway for growth.")

    # Team size
    if emp > 200:
        insights.append(f"👥 Large team ({emp} employees) suggests operational scale and execution capability.")
    elif emp < 10:
        insights.append(f"👥 Small team ({emp} employees) — lean but may face execution challenges at scale.")

    # Top-tier investor
    if top_inv:
        insights.append("⭐ Backed by a top-tier investor — strong signal of credibility and network access.")
    else:
        insights.append("📌 No top-tier investor backing — fundraising credibility may be limited.")

    # Revenue proxy
    rev_labels = {0:"No revenue",1:"Pre-revenue",2:"Early revenue",3:"Growing revenue",4:"Stable revenue",5:"Strong revenue"}
    insights.append(f"📈 Revenue stage: {rev_labels.get(int(rev), 'Unknown')} — affects sustainability outlook.")

    # Milestones
    if mile > 10:
        insights.append(f"🏁 {mile} milestones achieved — strong execution track record.")
    elif mile < 3:
        insights.append(f"⚠️ Only {mile} milestones — early stage with limited proven traction.")

    # Market size
    mkt_labels = {1:"Small (niche)", 2:"Medium", 3:"Large (scalable)"}
    insights.append(f"🌍 Targeting a {mkt_labels.get(int(mkt),'unknown')} market — impacts ceiling for growth.")

    # Founder experience
    if exp >= 4:
        insights.append(f"🎓 High founder experience (score {exp}/5) — strong leadership signal.")
    elif exp <= 2:
        insights.append(f"🎓 Limited founder experience (score {exp}/5) — may need senior hires.")

    # Patent
    if patent:
        insights.append("🔬 Patent held — provides competitive moat and IP protection.")

    # Company age
    if age > 10:
        insights.append(f"⏳ Established company ({age} years) — resilience tested over time.")
    elif age < 3:
        insights.append(f"🚀 Early-stage company ({age} years) — high upside but high uncertainty.")

    # Probability context
    insights.append(
        f"🤖 ANN confidence: {probs[status]*100:.1f}% for '{status}'. "
        f"Secondary: {sorted([(k,v) for k,v in probs.items() if k!=status], key=lambda x:-x[1])[0][0]} "
        f"({sorted([(k,v) for k,v in probs.items() if k!=status], key=lambda x:-x[1])[0][1]*100:.1f}%)"
    )
    return insights


def generate_recommendations(inp: dict, status: str) -> List[str]:
    """Generate actionable recommendations based on inputs and predicted status."""
    recs = []
    funding = inp.get("funding_total_usd_M", 0)
    rounds  = inp.get("num_funding_rounds", 1)
    emp     = inp.get("num_employees", 1)
    rev     = inp.get("revenue_proxy", 0)
    top_inv = inp.get("top_tier_investor", 0)
    exp     = inp.get("founder_experience", 1)
    patent  = inp.get("has_patent", 0)
    mkt     = inp.get("market_size", 1)
    mile    = inp.get("num_milestones", 0)
    rel     = inp.get("num_relationships", 0)
    cat     = inp.get("category", "")
    country = inp.get("country", "")

    # Status-specific header recommendation
    if status == "operating":
        recs.append("✅ Strong fundamentals detected. Focus on scaling and moat-building.")
    elif status == "acquired":
        recs.append("🔵 Acquisition signals present. Negotiate from strength and evaluate strategic buyers early.")
    else:
        recs.append("🚨 High risk signals. Immediate pivot or restructuring required to improve survival odds.")

    # Funding recommendations
    if funding < 5 and rounds < 2:
        recs.append("💡 Raise your next funding round soon. Target angel investors or seed-stage VCs to extend runway.")
    if not top_inv:
        recs.append("🤝 Target at least one top-tier VC/accelerator (Y Combinator, Sequoia, etc.) to boost credibility.")

    # Revenue
    if rev < 2:
        recs.append("📊 Prioritize reaching product-market fit and generating early revenue — reduces dependency on funding.")
    if rev >= 3:
        recs.append("💹 Solid revenue traction — consider expanding sales channels or entering adjacent markets.")

    # Team
    if emp < 20:
        recs.append("👥 Scale your team strategically — hire key functional leaders (CTO, CMO, CFO) to reduce single-point-of-failure risk.")
    if exp <= 2:
        recs.append("🎓 Bring on experienced advisors or co-founders — founder experience significantly impacts survival probability.")

    # IP / Patent
    if not patent and cat in ["Biotech", "Hardware", "AI/ML", "HealthTech"]:
        recs.append("🔬 File patents for core technology — IP protection is critical in your industry for competitive advantage.")

    # Market
    if mkt == 1:
        recs.append("🌍 Small market detected — consider pivoting or expanding to adjacent, larger markets to increase TAM.")

    # Milestones & Relationships
    if mile < 5:
        recs.append("🏁 Set and track 3–5 clear milestones for the next 12 months — traction metrics attract investors.")
    if rel < 3:
        recs.append("🤝 Grow strategic partnerships (distributors, tech partners, enterprise clients) to strengthen ecosystem position.")

    # Country-specific
    if country in ["USA", "India", "Germany", "UK", "Singapore"]:
        recs.append(f"🌐 You're in a strong startup ecosystem ({country}) — actively leverage local accelerators, grants, and networks.")

    return recs


if __name__ == "__main__":
    arts = load_artifacts()
    sample = {
        "company_age_years": 5,
        "category": "SaaS",
        "country": "India",
        "funding_round": "Series A",
        "funding_total_usd_M": 3.5,
        "num_funding_rounds": 2,
        "num_employees": 45,
        "num_investors": 4,
        "num_milestones": 7,
        "num_relationships": 5,
        "top_tier_investor": 0,
        "revenue_proxy": 2,
        "market_size": 2,
        "has_patent": 0,
        "founder_experience": 3
    }
    result = predict(sample, arts)
    print(f"\nPrediction: {result['status_emoji']} {result['predicted_status'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print("\nKey Insights:")
    for i in result["key_insights"]: print(" •", i)
    print("\nRecommendations:")
    for r in result["recommendations"]: print(" •", r)
