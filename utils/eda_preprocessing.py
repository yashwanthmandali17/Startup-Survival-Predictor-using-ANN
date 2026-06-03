"""
eda_preprocessing.py
Full EDA + preprocessing pipeline for the startup dataset.
Returns cleaned df, encoders, scaler, feature list, label encoder.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings, os
warnings.filterwarnings("ignore")

NUMERIC_COLS = [
    "company_age_years", "funding_total_usd_M", "num_funding_rounds",
    "num_employees", "num_investors", "num_milestones", "num_relationships",
    "top_tier_investor", "revenue_proxy", "market_size",
    "has_patent", "founder_experience"
]

CATEGORICAL_COLS = ["category", "country", "funding_round"]
TARGET_COL = "status"
DROP_COLS = ["company_id", "founded_year"]  # keep age, drop raw year

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDA_DIR = os.path.join(BASE_DIR, "data", "eda_plots")
os.makedirs(EDA_DIR, exist_ok=True)


def load_and_eda(csv_path: str, save_plots: bool = True, plot_dir: str = None):
    """Run EDA and return the raw dataframe with insights."""
    global EDA_DIR
    if plot_dir is not None:
        EDA_DIR = plot_dir
        os.makedirs(EDA_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTarget Distribution:\n{df[TARGET_COL].value_counts()}")
    print(f"\nNumerics:\n{df[NUMERIC_COLS].describe().T[['mean','std','min','max']].round(2)}")

    if save_plots:
        # 1. Status distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df[TARGET_COL].value_counts().plot(kind="bar", ax=axes[0], color=["#4CAF50","#2196F3","#F44336"])
        axes[0].set_title("Company Status Distribution")
        axes[0].set_xlabel("Status"); axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=0)

        df[TARGET_COL].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.1f%%",
                                           colors=["#4CAF50","#2196F3","#F44336"])
        axes[1].set_title("Status Proportion"); axes[1].set_ylabel("")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/01_status_distribution.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 2. Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 9))
        corr = df[NUMERIC_COLS].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/02_correlation_heatmap.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 3. Funding by status
        fig, ax = plt.subplots(figsize=(9, 5))
        df.groupby("status")["funding_total_usd_M"].median().sort_values().plot(
            kind="barh", ax=ax, color=["#F44336","#2196F3","#4CAF50"])
        ax.set_title("Median Funding (USD M) by Status")
        ax.set_xlabel("Median Funding (USD M)")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/03_funding_by_status.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 4. Category distribution by status
        fig, ax = plt.subplots(figsize=(14, 6))
        ct = pd.crosstab(df["category"], df["status"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
        ax.set_title("Status Distribution by Industry Category (%)")
        ax.set_xlabel("Category"); ax.set_ylabel("Percentage")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/04_category_status.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 5. Feature histograms
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        for i, col in enumerate(NUMERIC_COLS):
            df[col].hist(ax=axes[i], bins=25, color="#5C6BC0", edgecolor="white")
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
        plt.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/05_feature_distributions.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 6. Boxplot: key features vs status
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        key_feats = ["funding_total_usd_M", "num_employees", "company_age_years"]
        colors = {"operating": "#4CAF50", "acquired": "#2196F3", "closed": "#F44336"}
        for ax, feat in zip(axes, key_feats):
            for status, grp in df.groupby("status"):
                ax.boxplot(grp[feat], positions=[list(colors.keys()).index(status)],
                           widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=colors[status], alpha=0.7))
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(list(colors.keys()))
            ax.set_title(feat); ax.set_ylabel(feat)
        plt.suptitle("Key Features vs Company Status")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/06_boxplots.png", dpi=100, bbox_inches="tight")
        plt.close()

        print(f"\n✓ EDA plots saved to {EDA_DIR}/")

    return df


def preprocess(df: pd.DataFrame):
    """Clean, encode, and scale the dataframe. Returns X, y, scaler, encoders, feature_names."""
    df = df.copy()

    # Drop irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Fill numeric missing values with median
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = num_imputer.fit_transform(df[NUMERIC_COLS])

    # Fill categorical missing values with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    feature_names = list(X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n✓ Preprocessing done. Features: {len(feature_names)}, Samples: {len(y)}")
    print(f"  Classes: {dict(zip(target_le.classes_, np.bincount(y)))}")

    return X_scaled, y, scaler, label_encoders, target_le, feature_names


if __name__ == "__main__":
    df = load_and_eda("D:/try-2/data/startup_data.csv")
    X, y, scaler, le_dict, target_le, feats = preprocess(df)
    print("\nFeatures used:", feats)
    print("X shape:", X.shape, "| y shape:", y.shape)
