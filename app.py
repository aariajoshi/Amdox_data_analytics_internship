import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NeuralRetail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ NeuralRetail Analytics Dashboard")
st.markdown("AI-Driven Retail Analytics System")

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #dbeafe
    );
}

[data-testid="stSidebar"] {
    background-color: #e0f2fe;
}

h1,h2,h3 {
    color: #1e3a8a;
}

[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_feature_engineered_data.csv")

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df

df = load_data()




# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

selected_country = st.sidebar.multiselect(
    "Country",
    sorted(df["Country"].dropna().unique()),
    default=sorted(df["Country"].dropna().unique())
)

filtered_df = df[
    df["Country"].isin(selected_country)
]


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sales Dashboard",
        "Customer Segmentation",
        "Churn Analysis",
        "Business Insights"
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Overview":

    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Countries", df["Country"].nunique())

    with col3:
        st.metric(
            "Total Revenue",
            f"${df['TotalPrice'].sum():,.0f}"
        )

    with col4:
        st.metric(
            "Customers",
            int(df["Customer ID"].nunique())
        )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    avg_order = round(
        filtered_df["TotalPrice"].mean(),
        2
    )

    st.success(
        f"Average Order Value: ${avg_order}"
    )

# --------------------------------------------------
# SALES DASHBOARD
# --------------------------------------------------

elif page == "Sales Dashboard":

    st.header("📈 Sales Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top Countries")

        country_sales = (
            df.groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        country_sales.plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:

        st.subheader("Monthly Revenue")

        monthly_sales = (
            df.groupby("Month")["TotalPrice"]
            .sum()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        monthly_sales.plot(marker="o", ax=ax)
        st.pyplot(fig)

        st.subheader("Top 10 Products")

        top_products = (
            filtered_df.groupby("Description")
            ["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8,5))

        top_products.plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------
elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    st.subheader("RFM Features")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        data=df,
        x="PurchaseFrequency",
        y="MonetaryValue",
        alpha=0.5,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Purchase Frequency Distribution")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["PurchaseFrequency"], bins=40, ax=ax)
    st.pyplot(fig)

    st.subheader("Customer Value Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        x=filtered_df["MonetaryValue"],
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# CHURN ANALYSIS
# --------------------------------------------------
elif page == "Churn Analysis":

    st.header("⚠️ Churn Analysis")

    churn_threshold = 100

    churn_customers = df[df["Recency"] > churn_threshold]
    active_customers = df[df["Recency"] <= churn_threshold]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Potential Churn Customers",
            len(churn_customers)
        )

    with col2:
        st.metric(
            "Active Customers",
            len(active_customers)
        )

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        [len(active_customers), len(churn_customers)],
        labels=["Active", "Churn Risk"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
elif page == "Business Insights":

    st.header("💡 Business Insights")

    top_country = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .idxmax()
    )

    top_product = (
        df.groupby("Description")["Quantity"]
        .sum()
        .idxmax()
    )

    st.success(
        f"Highest Revenue Country: {top_country}"
    )

    st.success(
        f"Best Selling Product: {top_product}"
    )

    st.info(
        "Customers with high Purchase Frequency and high Monetary Value are premium customers."
    )

    st.info(
        "Customers with high Recency values may require retention campaigns."
    )

    st.info(
        "Customer segmentation can improve targeted marketing strategies."
    )

    st.subheader("🤖 AI Recommendations")

    st.info(
        "Focus marketing campaigns on high-value customers with high Monetary Value."
    )

    st.info(
        "Customers with high Recency scores should receive retention offers."
    )

    st.info(
        "Top-performing countries should receive additional inventory allocation."
    )

    st.info(
        "Premium customers can be targeted with loyalty programs."
    )