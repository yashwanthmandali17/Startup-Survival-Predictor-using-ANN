"""
generate_dataset.py
Generates a realistic Crunchbase-style startup/company dataset
with ~2000 rows and saves it to startup_data.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
N = 2000

CATEGORIES = [
    "Software", "Biotech", "E-Commerce", "FinTech", "EdTech",
    "HealthTech", "AI/ML", "CleanTech", "SaaS", "Hardware",
    "Media", "Gaming", "Logistics", "FoodTech", "AgriTech"
]

COUNTRIES = [
    "USA", "India", "UK", "Germany", "China", "Canada",
    "France", "Israel", "Singapore", "Brazil", "Australia",
    "Japan", "South Korea", "Netherlands", "Sweden"
]

FUNDING_ROUNDS = ["Seed", "Series A", "Series B", "Series C", "Series D+", "IPO"]

def generate_company(i):
    category = np.random.choice(CATEGORIES)
    country = np.random.choice(COUNTRIES)
    founded_year = np.random.randint(2000, 2022)
    company_age = 2024 - founded_year

    # Funding logic: more rounds → more funding
    round_idx = np.random.choice(len(FUNDING_ROUNDS), p=[0.25, 0.28, 0.20, 0.12, 0.08, 0.07])
    funding_round = FUNDING_ROUNDS[round_idx]
    base_funding = [0.3, 2.5, 12.0, 40.0, 120.0, 300.0][round_idx]
    funding_total_usd = max(0, np.random.lognormal(np.log(base_funding + 0.1), 0.7))

    num_funding_rounds = round_idx + 1 + np.random.randint(0, 2)

    # Employees
    employee_base = [5, 25, 80, 200, 500, 1500][round_idx]
    num_employees = max(1, int(np.random.lognormal(np.log(employee_base), 0.6)))

    # Investors
    num_investors = max(0, round_idx * 2 + np.random.randint(0, 5))

    # Milestones
    num_milestones = max(0, int(company_age * 0.7 + np.random.randint(-2, 5)))

    # Key relationships (partnerships, advisors)
    num_relationships = max(0, int(num_investors * 0.8 + np.random.randint(0, 4)))

    # Has top-tier investor flag
    top_tier_investor = int(np.random.random() < (0.05 + round_idx * 0.08))

    # Revenue (proxy 0-5 scale)
    revenue_proxy = min(5, max(0, round(round_idx * 0.9 + np.random.normal(0, 0.5), 1)))

    # Market size (1=small, 2=medium, 3=large)
    market_size = np.random.choice([1, 2, 3], p=[0.25, 0.45, 0.30])

    # Has patent
    has_patent = int(np.random.random() < 0.15 + 0.05 * (category in ["Biotech", "Hardware", "AI/ML"]))

    # Team size quality (founder experience proxy 1-5)
    founder_experience = np.random.randint(1, 6)

    # --- Status label (target) ---
    # Score-based status determination
    score = (
        round_idx * 1.5
        + num_milestones * 0.08
        + top_tier_investor * 1.2
        + revenue_proxy * 0.5
        + founder_experience * 0.3
        + market_size * 0.4
        + (company_age * 0.05)
        + np.random.normal(0, 0.8)
    )

    if score > 5.5:
        status = "operating"
    elif score > 3.5:
        # mid-range: could be acquired or operating
        status = np.random.choice(["operating", "acquired"], p=[0.65, 0.35])
    elif score > 2.0:
        status = np.random.choice(["operating", "closed", "acquired"], p=[0.40, 0.45, 0.15])
    else:
        status = np.random.choice(["closed", "operating"], p=[0.75, 0.25])

    return {
        "company_id": f"C{i:04d}",
        "founded_year": founded_year,
        "company_age_years": company_age,
        "category": category,
        "country": country,
        "funding_round": funding_round,
        "funding_total_usd_M": round(funding_total_usd, 2),
        "num_funding_rounds": num_funding_rounds,
        "num_employees": num_employees,
        "num_investors": num_investors,
        "num_milestones": num_milestones,
        "num_relationships": num_relationships,
        "top_tier_investor": top_tier_investor,
        "revenue_proxy": revenue_proxy,
        "market_size": market_size,
        "has_patent": has_patent,
        "founder_experience": founder_experience,
        "status": status
    }

if __name__ == "__main__":
    import os
    records = [generate_company(i) for i in range(N)]
    df = pd.DataFrame(records)
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup_data.csv")
    df.to_csv(OUT, index=False)
    print(f"Dataset created: {df.shape}")
    print(df["status"].value_counts())
    print(df.head(3))
