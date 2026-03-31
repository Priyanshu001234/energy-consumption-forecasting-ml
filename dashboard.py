
# ==============================
# IMPORTS – libraries for data, ML, plots and dashboard
# ==============================
import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,   
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ==============================
# STREAMLIT PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Predictive Analytics – Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# FUNCTION 1 – Load all region CSV files
# (Unit I: Data preparation)
# ==============================
@st.cache_data
def load_region_data(data_path="data"):
    # pattern for all CSVs like AEP_hourly.csv, PJME_hourly.csv, etc.
    pattern = os.path.join(data_path, "*_hourly.csv")
    files = glob.glob(pattern)

    regions = []      # list of region names
    data_dict = {}    # dict: region -> dataframe

    for file in files:
        df = pd.read_csv(file)

        # first column is datetime, second is MW
        dt_col = df.columns[0]
        val_col = df.columns[1]

        df = df[[dt_col, val_col]]
        df.columns = ["Datetime", "MW"]

        # convert to datetime and sort
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime")

        # region name from filename
        region = os.path.basename(file).replace("_hourly.csv", "").replace(".csv", "")
        regions.append(region)
        data_dict[region] = df

    regions = sorted(regions)
    return regions, data_dict


# ==============================
# FUNCTION 2 – Build daily multi-region dataset
# (Used later for clustering / PCA)
# ==============================
@st.cache_data
def build_cluster_data(data_dict):
    merged = None

    # merge all regions on Datetime
    for region, df in data_dict.items():
        temp = df[["Datetime", "MW"]].copy()
        temp = temp.rename(columns={"MW": region})  # one column per region

        if merged is None:
            merged = temp
        else:
            merged = pd.merge(merged, temp, on="Datetime", how="outer")

    if merged is None:
        return None

    # set index and resample to daily mean
    merged = merged.sort_values("Datetime").set_index("Datetime")
    daily = merged.resample("D").mean()
    daily = daily.dropna(how="all")
    return daily


# ==============================
# FUNCTION 3 – Feature engineering
# (Unit II & III: supervised learning features)
# ==============================
def create_features(df):
    df = df.copy()
    df = df.sort_values("Datetime")

    # time-based features
    df["hour"] = df["Datetime"].dt.hour
    df["day"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month
    df["year"] = df["Datetime"].dt.year
    df["dow"] = df["Datetime"].dt.dayofweek   # day of week (0–6)

    # lag and rolling mean features (time-series context)
    df["lag_1"] = df["MW"].shift(1)
    df["rolling_24"] = df["MW"].rolling(24).mean()

    df = df.dropna()

    feature_cols = ["hour", "day", "month", "year", "dow", "lag_1", "rolling_24"]
    return df, feature_cols


# ==============================
# LOAD DATASETS
# ==============================
regions, data_dict = load_region_data()
if not regions:
    st.error("No hourly CSV files found in data folder.")
    st.stop()

# daily dataset across all regions (for clustering/PCA)
cluster_daily = build_cluster_data(data_dict)

# ==============================
# STYLING AND HEADER (UI only)
# ==============================
st.markdown(
    """
    <style>
    /* Background feel */
    .main {
        background: radial-gradient(circle at top left,
    #ff512f 0%,
    #dd2476 20%,
    #93329e 40%,
    #3478f6 60%,
    #00e5ff 80%,
    #4ade80 100%
);

    }

    .big-title {
        font-size: 40px;
        font-weight: 800;
        background: -webkit-linear-gradient(90deg,#38bdf8,#a855f7,#f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 16px;
        color: #e5e7eb;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .unit-pill {
        display:inline-block;
        padding:4px 12px;
        border-radius:999px;
        background:linear-gradient(90deg,#0ea5e9,#6366f1);
        font-size:12px;
        margin-right:8px;
        color:#f9fafb;
        box-shadow:0 0 12px rgba(59,130,246,0.45);
    }

    .feature-box {
        background:linear-gradient(135deg,#1e293b,#020617);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.6);
        font-size: 14px;
        color: #e5e7eb;
        height: 130px;
        box-shadow:0 18px 35px rgba(15,23,42,0.9);
    }

    .feature-title {
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 6px;
        color:#f97316;
        letter-spacing:0.03em;
        text-transform:uppercase;
    }

    .stMetric {
        background: radial-gradient(circle at top,#0f172a,#020617);
        padding: 12px 16px;
        border-radius: 16px;
        border: 1px solid rgba(148,163,184,0.6);
        box-shadow:0 12px 30px rgba(15,23,42,0.9);
    }
    .stMetric label {
        color:#cbd5f5 !important;
        font-size:13px;
        text-transform:uppercase;
        letter-spacing:0.06em;
    }
    .stMetric [data-testid="stMetricValue"] {
        color:#4ade80 !important;
        font-size:28px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# project title + syllabus mapping
st.markdown('<p class="big-title">Predictive Analytics – Hourly Energy Consumption Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">End-to-end project covering data preparation, supervised learning, unsupervised learning, dimensionality reduction, and model evaluation on PJM hourly energy data.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="unit-pill"> Data Preparation & EDA</span>'
    '<span class="unit-pill"> Regression</span>'
    '<span class="unit-pill"> Classification</span>'
    '<span class="unit-pill"> Clustering</span>'
    '<span class="unit-pill"> PCA</span>'
    '<span class="unit-pill"> Model Performance</span>',
    unsafe_allow_html=True,
)

# 3 small description cards
f1, f2, f3 = st.columns(3)
f1.markdown(
    '<div class="feature-box"><div class="feature-title">Business Goal</div>Forecast regional electricity demand and study temporal usage patterns for better grid planning.</div>',
    unsafe_allow_html=True,
)
f2.markdown(
    '<div class="feature-box"><div class="feature-title">ML Scope</div>Apply regression, classification, clustering, and PCA on real-world time-series data.</div>',
    unsafe_allow_html=True,
)
f3.markdown(
    '<div class="feature-box"><div class="feature-title">Deliverables</div>Interactive dashboard, evaluation metrics, visual insights, and model comparison for academic & professional use.</div>',
    unsafe_allow_html=True,
)

# ==============================
# SIDEBAR – REGION & DATE FILTERS
# ==============================
with st.sidebar:
    st.title("Mayank Sharma")

    # choose region
    selected_region = st.selectbox("Region", regions)
    df_region = data_dict[selected_region].copy()

    # choose date range
    min_date = df_region["Datetime"].min().date()
    max_date = df_region["Datetime"].max().date()
    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    # aggregation level
    freq_label = st.selectbox("Aggregation", ["Hourly", "Daily", "Weekly"])

# validate dates
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# filter current region data by date range
mask = (df_region["Datetime"].dt.date >= start_date) & (df_region["Datetime"].dt.date <= end_date)
df_range = df_region.loc[mask].copy()

if df_range.empty:
    st.warning("No data in this range. Try different dates.")
    st.stop()

# allow user to download filtered data
csv_bytes = df_range.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_bytes,
    file_name=f"{selected_region}_filtered_data.csv",
    mime="text/csv",
)

# ==============================
# AGGREGATION (Hourly / Daily / Weekly)
# ==============================
if freq_label == "Hourly":
    df_ts = df_range.set_index("Datetime")
elif freq_label == "Daily":
    df_ts = df_range.set_index("Datetime").resample("D").mean()
elif freq_label == "Weekly":
    df_ts = df_range.set_index("Datetime").resample("W").mean()
else:
    df_ts = df_range.set_index("Datetime")

# ==============================
# KPI METRICS + CHANGE vs PREVIOUS PERIOD
# ==============================
avg_load = df_ts["MW"].mean()
max_load = df_ts["MW"].max()
min_load = df_ts["MW"].min()

# simple previous period comparison based on same length window before start date
period_length = df_ts.index.max() - df_ts.index.min()
prev_start = df_ts.index.min() - period_length - timedelta(seconds=1)
prev_end = df_ts.index.min() - timedelta(seconds=1)
df_prev = df_region.set_index("Datetime").loc[prev_start:prev_end]

if not df_prev.empty:
    prev_avg = df_prev["MW"].mean()
else:
    prev_avg = df_region["MW"].mean()

change_pct = ((avg_load - prev_avg) / prev_avg) * 100

k1, k2, k3 = st.columns(3)
k1.metric("Average MW", f"{avg_load:,.0f}")
k2.metric("Peak MW", f"{max_load:,.0f}")
k3.metric("Lowest MW", f"{min_load:,.0f}")

if change_pct > 0:
    kpi_change = f"⬆️ {change_pct:.2f}%"
else:
    kpi_change = f"⬇️ {abs(change_pct):.2f}%"

st.markdown(f"**Change vs previous period:** {kpi_change}")

# ==============================
# TABS – link to syllabus units
# ==============================
tab_overview, tab_viz, tab_reg, tab_clf, tab_cluster, tab_eval = st.tabs(
    [
        "Overview & EDA",
        "Advanced Visuals",
        "Regression",
        "Classification",
        "Clustering & PCA",
        "Model Performance",
    ]
)

# -------------------------------------------------
# TAB 1 – OVERVIEW & EDA  (Unit I)
# -------------------------------------------------
with tab_overview:
    st.subheader(" Exploratory Data Analysis")

    # 📌 Project Objectives Display Box
    st.markdown("""
    <style>
    .obj-box {
        background:linear-gradient(120deg,#fef9c3,#fee2e2,#e0f2fe);
        border-radius:12px;
        padding:15px;
        font-size:14px;
        border:1px solid #d4d4d4;
        box-shadow:0 5px 12px rgba(0,0,0,0.12);
        margin-bottom:15px;
    }
    .obj-box b{
        color:#1d4ed8;
    }
    </style>

    <div class="obj-box">
    <b>🎯 Project Objectives</b><br><br>
    1. <b>Analyse electricity consumption behaviour</b> across time scales (hourly, daily, seasonal).<br><br>
    2. <b>Predict demand trends</b> using regression to support planning and forecasting.<br><br>
    3. <b>Detect peak load periods</b> through classification models for grid alerting.<br><br>
    4. <b>Identify hidden demand clusters</b> using unsupervised learning and PCA.<br><br>
    5. <b>Validate models</b> using performance metrics and cross-validation for reliability.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        fig_ts = px.line(
            df_ts.reset_index(),
            x="Datetime",
            y="MW",
            labels={"Datetime": "Time", "MW": "MW"},
            title=f"{selected_region} – {freq_label} consumption trend",
        )
        fig_ts.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
        sns.histplot(df_ts["MW"], bins=30, ax=ax_hist, kde=True)
        ax_hist.set_xlabel("MW")
        ax_hist.set_title("Distribution of load values")
        st.pyplot(fig_hist)

    df_feat_over = df_range.copy()
    df_feat_over["hour"] = df_feat_over["Datetime"].dt.hour
    df_feat_over["month"] = df_feat_over["Datetime"].dt.month
    c3, c4 = st.columns(2)

    with c3:
        hourly_avg = df_feat_over.groupby("hour")["MW"].mean()
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        hourly_avg.plot(kind="bar", ax=ax_h)
        ax_h.set_xlabel("Hour of day")
        ax_h.set_ylabel("Average MW")
        ax_h.set_title("Average load by hour")
        st.pyplot(fig_h)

    with c4:
        monthly_avg = df_feat_over.groupby("month")["MW"].mean()
        fig_m, ax_m = plt.subplots(figsize=(5, 3))
        monthly_avg.plot(kind="bar", ax=ax_m)
        ax_m.set_xlabel("Month")
        ax_m.set_ylabel("Average MW")
        ax_m.set_title("Average load by month")
        st.pyplot(fig_m)

# -------------------------------------------------
# TAB 2 – ADVANCED VISUALS  (extra EDA)
# -------------------------------------------------
with tab_viz:
    st.subheader("Advanced Visualizations – All-in-one View")

    # use engineered features for some charts
    df_feat_full, feature_cols_full = create_features(df_region)
    df_feat_full = df_feat_full[
        (df_feat_full["Datetime"].dt.date >= start_date)
        & (df_feat_full["Datetime"].dt.date <= end_date)
    ]

    col1, col2 = st.columns(2)

    # rolling daily area chart
    with col1:
        st.markdown("**1. Daily Area Chart (7-day Rolling Trend)**")
        daily = df_region.set_index("Datetime").resample("D").mean()
        daily = daily[(daily.index.date >= start_date) & (daily.index.date <= end_date)]
        if len(daily) > 0:
            daily["rolling_7"] = daily["MW"].rolling(7).mean()
            fig_area = px.area(
                daily.reset_index(),
                x="Datetime",
                y="rolling_7",
                labels={"rolling_7": "MW", "Datetime": "Date"},
            )
            fig_area.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("No daily data available for this range.")

    # hourly boxplot
    with col2:
        st.markdown("**2. Boxplot – Hourly Load Distribution**")
        df_box = df_region[
            (df_region["Datetime"].dt.date >= start_date)
            & (df_region["Datetime"].dt.date <= end_date)
        ].copy()
        df_box["hour"] = df_box["Datetime"].dt.hour
        if len(df_box) > 0:
            fig_box, ax_box = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=df_box, x="hour", y="MW", ax=ax_box)
            ax_box.set_xlabel("Hour of day")
            ax_box.set_ylabel("MW")
            ax_box.set_title("Distribution of load by hour")
            st.pyplot(fig_box)
        else:
            st.info("Not enough data to display boxplot.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    # scatter lag vs current (autocorrelation)
    with col3:
        st.markdown("**3. Scatter – Current vs Lagged Load (Autocorrelation)**")
        if not df_feat_full.empty:
            fig_sc, ax_sc = plt.subplots(figsize=(5, 3))
            ax_sc.scatter(df_feat_full["lag_1"], df_feat_full["MW"], alpha=0.4, s=10)
            ax_sc.set_xlabel("Previous hour MW (lag_1)")
            ax_sc.set_ylabel("Current hour MW")
            ax_sc.set_title("Relationship between consecutive hours")
            st.pyplot(fig_sc)
        else:
            st.info("Not enough data for scatter plot.")

    # correlation heatmap
    with col4:
        st.markdown("**4. Correlation Heatmap – Features vs Target**")
        if not df_feat_full.empty:
            corr_cols = feature_cols_full + ["MW"]
            corr = df_feat_full[corr_cols].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
            ax_corr.set_title("Correlation between engineered features and MW")
            st.pyplot(fig_corr)
        else:
            st.info("Not enough data for correlation heatmap.")

# -------------------------------------------------
# TAB 3 – REGRESSION (Unit II)
# -------------------------------------------------
with tab_reg:
    st.subheader(" Supervised Learning: Regression")

    df_feat_reg, feature_cols_reg = create_features(df_region)
    if len(df_feat_reg) < 200:
        st.warning("Not enough data for regression modelling.")
    else:
        # restrict to selected date range; if too small, use all
        df_feat_range = df_feat_reg[
            (df_feat_reg["Datetime"].dt.date >= start_date)
            & (df_feat_reg["Datetime"].dt.date <= end_date)
        ]
        if len(df_feat_range) < 200:
            df_feat_range = df_feat_reg

        X = df_feat_range[feature_cols_reg]
        y = df_feat_range["MW"]

        # simple time-based train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model_reg = LinearRegression()
        model_reg.fit(X_train, y_train)
        y_pred = model_reg.predict(X_test)

        # regression metrics from syllabus
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:,.2f}")
        c2.metric("RMSE", f"{rmse:,.2f}")
        c3.metric("MSE", f"{mse:,.2f}")
        c4.metric("R² score", f"{r2:,.3f}")

        # line plot of last 300 predictions vs actual
        st.write("Prediction vs Actual (last 300 samples)")
        n_plot = min(300, len(y_test))
        fig_reg, ax_reg = plt.subplots(figsize=(9, 3))
        ax_reg.plot(y_test.values[-n_plot:], label="Actual")
        ax_reg.plot(y_pred[-n_plot:], label="Predicted")
        ax_reg.legend()
        ax_reg.set_xlabel("Time index")
        ax_reg.set_ylabel("MW")
        st.pyplot(fig_reg)

# -------------------------------------------------
# TAB 4 – CLASSIFICATION (Unit III)
# -------------------------------------------------
with tab_clf:
    st.subheader(" Supervised Learning: Classification")

    df_feat_cls, feature_cols_cls = create_features(df_region)
    if len(df_feat_cls) < 200:
        st.warning("Not enough data for classification.")
    else:
        df_feat_range = df_feat_cls[
            (df_feat_cls["Datetime"].dt.date >= start_date)
            & (df_feat_cls["Datetime"].dt.date <= end_date)
        ]
        if len(df_feat_range) < 200:
            df_feat_range = df_feat_cls

        # target: high_load = 1 for top 25% consumption
        threshold = df_feat_range["MW"].quantile(0.75)
        df_feat_range["high_load"] = (df_feat_range["MW"] > threshold).astype(int)

        X = df_feat_range[feature_cols_cls]
        y = df_feat_range["high_load"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=120, max_depth=None)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # classification metrics required in syllabus
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1 score", f"{f1:.3f}")

        # confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix (High vs Low load)")
        st.pyplot(fig_cm)

# -------------------------------------------------
# TAB 5 – CLUSTERING & PCA (Unit IV & V)
# -------------------------------------------------
with tab_cluster:
    st.subheader(" Clustering and PCA (Unsupervised Learning)")

    # need data from at least 2 regions
    if cluster_daily is None or cluster_daily.shape[1] < 2:
        st.warning("Need data from at least 2 regions for clustering and PCA.")
    else:
        df_cluster = cluster_daily.copy()

        # fill missing values along time axis
        df_cluster = df_cluster.fillna(method="ffill").fillna(method="bfill")
        df_cluster = df_cluster.dropna(how="any")

        if df_cluster.shape[0] < 30:
            st.warning("Clustering requires at least 30 days of clean data across regions.")
        else:
            # scale features before k-means & PCA
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_cluster.values)   # shape: (days, regions)

            # K-Means clustering into 3 groups
            kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(scaled)

            df_cluster_res = df_cluster.copy()
            df_cluster_res["cluster"] = clusters

            st.write("Cluster size (number of days in each group):")
            st.write(df_cluster_res["cluster"].value_counts())

            # PCA to reduce multi-region data to 2 dimensions for plotting
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
            pca_df["cluster"] = clusters

            fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=pca_df,
                x="PC1",
                y="PC2",
                hue="cluster",
                palette="Set2",
                ax=ax_pca,
            )
            ax_pca.set_title("Daily Load Profiles – PCA + K-Means Clusters")
            st.pyplot(fig_pca)

# -------------------------------------------------
# TAB 6 – MODEL PERFORMANCE / CROSS-VALIDATION (Unit VI)
# -------------------------------------------------
with tab_eval:
    st.subheader(" Model Performance & Cross-Validation")

    df_feat_eval, feature_cols_eval = create_features(df_region)
    if len(df_feat_eval) < 300:
        st.warning("Not enough samples for cross-validation.")
    else:
        X = df_feat_eval[feature_cols_eval]
        y = df_feat_eval["MW"]

        # 5-fold K-Fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        reg = LinearRegression()

        # negative MSE (sklearn convention), convert to RMSE
        neg_mse_scores = cross_val_score(
            reg, X, y, cv=kf, scoring="neg_mean_squared_error"
        )
        rmse_scores = np.sqrt(-neg_mse_scores)

        st.write("5-Fold Cross-Validation – Linear Regression (Regression Task)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("RMSE per fold:", [f"{s:.2f}" for s in rmse_scores])
        with c2:
            st.metric("Average CV RMSE", f"{rmse_scores.mean():.2f}")

        fig_cv, ax_cv = plt.subplots(figsize=(5, 3))
        ax_cv.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker="o")
        ax_cv.set_xlabel("Fold")
        ax_cv.set_ylabel("RMSE")
        ax_cv.set_title("Bias–Variance View via Cross-Validation")
        st.pyplot(fig_cv)

    st.markdown(
        """
        **Interpretation notes for report / viva:**
        - Lower RMSE across folds indicates better generalisation.
        - Large variation between folds suggests higher variance.
        - Comparing training and CV error helps discussion of the bias–variance trade-off.
        """,
    )

# -------------------------------------------------
# FOOTER – project credit
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "Designed as a Predictive Analytics course project · "
    "Student: **Priyanshu Raj Sharma** · Dataset: PJM Hourly Energy Consumption"
)
