# Customer Segmentation MLOps Project — Mentor Explanation Guide

## 🎯 One-Liner Pitch
> "I built an end-to-end **Customer Segmentation Pipeline** using MLOps practices — from raw e-commerce data to actionable business segments, fully tracked in MLflow."

---

## 📌 Problem Statement

**Business Problem:** An e-commerce company has thousands of customers, but treats them all the same — same emails, same offers. This wastes marketing budget and loses customers.

**Solution:** Use machine learning to automatically **segment customers** into groups (Champions, At Risk, Lost, etc.) so the business can personalize marketing strategies per segment.

**Dataset:** UCI Online Retail Dataset — **541,909 transactions** across 38 countries, containing Invoice, StockCode, Description, Quantity, Price, Customer ID, and Country.

---

## 🏗️ Architecture — 6-Step Pipeline

```
Raw Data (CSV)
    │
    ▼
┌──────────────────────────────┐
│ Step 1: Data Exploration     │ → Missing values, data types, basic stats
│ Step 2: Data Preprocessing   │ → Cleaning, feature engineering, scaling
│ Step 3: EDA                  │ → 10 visualizations (trends, distributions)
│ Step 4: K-Means Clustering   │ → RFM Analysis → K-Means (k=4)
│ Step 5: Cluster Labeling     │ → Business labels + revenue/churn analysis
│ Step 6: Classification       │ → Random Forest to predict segments
└──────────────────────────────┘
    │
    ▼
MLflow (Experiment: "customer" — 6 runs, 23+ graphs, all metrics)
```

All steps are orchestrated via `run_pipeline.py` — one command runs everything.

---

## 📝 Step-by-Step Explanation

### Step 1: Data Exploration (`check_missing_values.py`)
**What:** Load raw data, check for missing values, data types, and basic statistics.

**Key findings to mention:**
- Dataset has **541,909 rows × 8 columns**
- Missing values found in `Customer ID` and `Description`
- Logged to MLflow: missing value counts, percentages, data types

**Why it matters:** "Before building any model, we need to understand the data quality."

---

### Step 2: Data Preprocessing (`data_preprocessing.py`)
**What:** Clean data and engineer new features.

**3 cleaning steps:**
1. **Drop rows** with missing Description
2. **Generate unique IDs** for missing Customer IDs (instead of dropping — preserves data)
3. **Remove duplicates**

**Feature Engineering:**
- Date features: `Year`, `Month`, `Day`, `Hour`, `DayOfWeek`
- Transaction features: `TotalPrice`, `TransactionSize`, `UniqueItems`
- Encoding: One-Hot for Country (38 → 38 binary columns)
- Scaling: StandardScaler on numeric features

**Mentor talking point:** *"I chose to fill missing Customer IDs instead of dropping rows because dropping would lose ~25% of the data. This is a design decision — I documented it in MLflow."*

---

### Step 3: EDA (`eda.py`)
**What:** 10 visualizations to understand business patterns.

| Graph | Key Insight to Mention |
|:------|:----------------------|
| Monthly Sales Trend | Identify seasonal peaks (e.g., November/December spike) |
| Top 10 Countries | UK dominates revenue — important for market focus |
| Top 10 Products | Shows which products drive volume |
| Hourly Heatmap | Peak hours = 10am–2pm on weekdays — useful for email timing |
| Revenue Distribution | Most transactions are small; a few are very large (Pareto) |
| Sales by Day of Week | Thursday is the best sales day |
| Customer Purchase Frequency | Most customers buy only 1–2 times (opportunity) |
| Monthly Active Customers | Customer base growth/decline over time |
| Quantity vs Price | Shows pricing patterns and outliers |
| Top 10 Customers | Just 10 customers contribute significant revenue |

---

### Step 4: K-Means Clustering (`train_model.py`)
**What:** Core ML step — RFM Analysis + K-Means clustering.

**RFM Analysis (explain this clearly):**
- **R**ecency: Days since last purchase (lower = better)
- **F**requency: Number of orders (higher = better)
- **M**onetary: Total spend (higher = better)

**Why K-Means:**
- Simple, interpretable, works well for customer segmentation
- Features must be scaled (StandardScaler) because K-Means is distance-based

**How I chose k=4:**
- **Elbow Curve:** Tested k=2 to k=10, found the "elbow" at k=4
- **Silhouette Score:** Validated cluster quality

**3 clustering metrics:**
| Metric | What It Measures | Ideal |
|:-------|:----------------|:------|
| Silhouette Score | Cluster quality | Closer to 1 |
| Davies-Bouldin Index | Cluster separation | Lower is better |
| Calinski-Harabasz | Cluster density | Higher is better |

**Mentor talking point:** *"I tested multiple k values — the elbow curve and silhouette score both suggested k=4 is optimal. This is logged in MLflow so anyone can verify."*

---

### Step 5: Cluster Labeling (`analyze_clusters.py`)
**What:** Convert numeric clusters into business-meaningful segments.

**Labeling logic:** Rank each cluster by R, F, M scores → assign labels:

| Segment | Characteristics | Business Action |
|:--------|:---------------|:----------------|
| **Champions** | Low recency, high frequency, high monetary | VIP perks, loyalty rewards |
| **Potential Loyalists** | Medium across all metrics | Nurture with targeted offers |
| **At Risk** | High recency, were once active | "We miss you" campaigns |
| **Lost** | Very high recency, low everything | Win-back discount or stop spending |

**Business Metrics:**
- Revenue share per segment (Champions likely contribute 60%+)
- Churn probability (Lost segment → close to 1.0)

**Mentor talking point:** *"This is where ML meets business value. The segments directly translate to marketing strategies that save money."*

---

### Step 6: Random Forest Classification (`train_classifier.py`)
**What:** Train a supervised model to predict segments for **new customers**.

**Why this step exists:**
> "K-Means is great for discovery, but in production you can't re-cluster every time. The Random Forest learns the pattern so it can instantly classify new customers."

**Features used:**
- AvgBasketValue, AvgBasketSize, Volatility, FavoriteDay, Country

**Results (mention these numbers):**
| Metric | Score |
|:-------|:------|
| Accuracy | ~93% |
| F1 Score | ~0.93 |
| ROC-AUC | ~0.99 |

**Key insight — Feature Importance:**
- `AvgBasketValue` = **91%** importance (the #1 predictor)
- Country and Day = almost irrelevant

**Mentor talking point:** *"The model tells us that how much a customer spends is the single strongest predictor of their segment — not where they live or when they shop."*

---

## 🔧 MLOps Practices Used

| Practice | How I Used It |
|:---------|:-------------|
| **Experiment Tracking** | MLflow — every step logs params, metrics, artifacts |
| **Reproducibility** | `random_state=42` everywhere, parameters logged |
| **Pipeline Automation** | `run_pipeline.py` runs all 6 steps in one command |
| **Model Versioning** | Both K-Means and Random Forest models saved in MLflow |
| **Artifact Management** | 23+ graphs, CSVs, and reports stored as MLflow artifacts |
| **Feature Engineering** | RFM features, basket metrics, date features |

---

## 🛠️ Tech Stack

| Category | Tool |
|:---------|:-----|
| Language | Python 3 |
| ML | scikit-learn (KMeans, RandomForest) |
| MLOps | MLflow (tracking, model logging, artifacts) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Dataset | UCI Online Retail (541K transactions) |

---

## 💬 Common Mentor Questions & Answers

**Q: Why K-Means and not DBSCAN or hierarchical?**
> K-Means is simpler, interpretable, and works well for RFM features. I tested the cluster quality with 3 different metrics.

**Q: Why Random Forest for classification?**
> It handles mixed feature types, provides feature importance, and is robust without much tuning. Accuracy is 93%+ out-of-the-box.

**Q: How would you deploy this?**
> The Random Forest model is saved in MLflow. Next step would be a Flask/FastAPI endpoint that takes customer data and returns the predicted segment.

**Q: What if the data changes over time?**
> The pipeline is fully automated — re-run `run_pipeline.py` with new data and compare runs in MLflow to detect model drift.

**Q: Why 4 segments and not 3 or 5?**
> The elbow curve and silhouette analysis both indicated k=4. I tested k=2 through k=10 and logged all results in MLflow.

---

## 🎤 Demo Flow (5-10 minutes)

1. **Open MLflow** → http://localhost:5000 → "customer" experiment
2. **Show the 6 runs** → explain the pipeline flow
3. **Click Step3_EDA** → show the 10 graphs (monthly trend, heatmap)
4. **Click Step4** → show the **elbow curve** (why k=4)
5. **Click Step5** → show **revenue share pie** and **churn probability**
6. **Click Step6** → show **confusion matrix** and **ROC curves**
7. **Run the pipeline live**: `python run_pipeline.py` → show it runs end-to-end
8. **Conclude** → "Every parameter, metric, and graph is tracked and reproducible"
