import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Healthcare Intelligence Dashboard",
    page_icon="🏥",
    layout="wide"

)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 95%;
}
</style>
""", unsafe_allow_html=True)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#f0fdfa,#ecfeff);
}

[data-testid="stSidebar"] {
    background-color: #dff7f3;
}

[data-testid="stSidebar"] label {
    color: #0f766e !important;
    font-weight: 600;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0f766e !important;
}

.block-container {
    max-width: 95%;
    padding-top: 1rem;
}

h1,h2,h3 {
    color: #0f766e;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #c7f0e8;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(r"C:\Users\Pranav\Downloads\Amdox internship\Medicine prediction\Medicine_prediction\Cleaned_Dataset.csv")

# =====================================
# HEADER
# =====================================

st.markdown("""
<h1 style='text-align:center;'>
🏥 Personalized Healthcare & Medicine Recommendation System
</h1>
""", unsafe_allow_html=True)

st.info(
    "AI-Powered Disease Analytics • Risk Assessment • Medicine Recommendation"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Patient Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    df["gender"].unique(),
    default=df["gender"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["risk_level"].isin(risk_filter))
]

patient_age = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (
        int(df["age"].min()),
        int(df["age"].max())
    )
)

filtered_df = filtered_df[
    (filtered_df["age"] >= patient_age[0]) &
    (filtered_df["age"] <= patient_age[1])
]

# =====================================
# KPI CARDS
# =====================================

patients = len(filtered_df)
diseases = filtered_df["disease"].nunique()
avg_age = round(filtered_df["age"].mean(),1)

high_risk = len(
    filtered_df[
        filtered_df["risk_level"] == "High"
    ]
)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Patients", patients)
col2.metric("Diseases", diseases)
col3.metric("Average Age", avg_age)
col4.metric("High Risk", high_risk)

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Analytics",
    "🩺 Risk Assessment",
    "💊 Medicine",
    "🔍 Disease Explorer",
    "🧬 Health Score",
    "📁 Dataset"
])

# =====================================
# ANALYTICS TAB
# =====================================

with tab1:

    st.subheader("Disease Distribution")

    top_diseases = (
        filtered_df["disease"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        x=top_diseases.values,
        y=top_diseases.index,
        ax=ax
    )

    st.pyplot(fig)

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Risk Level Distribution")

        fig, ax = plt.subplots()

        sns.countplot(
            data=filtered_df,
            x="risk_level",
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Gender Distribution")

        fig, ax = plt.subplots()

        filtered_df["gender"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        filtered_df["age"],
        bins=20,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# PIE CHART

    risk_counts = (
        filtered_df["risk_level"]
        .value_counts()
    )

    fig, ax = plt.subplots()

    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # TOP DISEASE

    st.subheader("🏆 Most Common Disease")

    top_disease = (
        filtered_df["disease"]
        .value_counts()
        .idxmax()
    )

    st.success(
        f"Most Common Disease: {top_disease}"
    )

    st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(
    include=["int64","float64"]
)

fig, ax = plt.subplots(
    figsize=(10,6)
)

sns.heatmap(
    numeric_df.corr(),
    cmap="BuGn",
    annot=True,
    ax=ax
)

st.pyplot(fig)

# =====================================
# RISK ASSESSMENT
# =====================================

with tab2:

    st.subheader("🩺 Patient Risk Assessment")

    age = st.slider(
    "Age",
    1,
    100,
    30,
    key="risk_age"
)

    bp = st.selectbox(
        "Blood Pressure",
        [1,2,3]
    )

    chol = st.selectbox(
        "Cholesterol Level",
        [1,2,3]
    )

    if st.button("Analyze Patient"):

        score = 0

        if age > 60:
            score += 1

        if bp == 3:
            score += 1

        if chol == 3:
            score += 1

        if score >= 2:

            st.error(
                "⚠️ High Risk Patient"
            )

        elif score == 1:

            st.warning(
                "🟡 Moderate Risk Patient"
            )

        else:

            st.success(
                "🟢 Low Risk Patient"
            )



# =====================================
# MEDICINE RECOMMENDATION
# =====================================

with tab3:

    st.subheader(
        "💊 Medicine Recommendation"
    )

    medicine_dict = {
        "Influenza":"Oseltamivir",
        "Common Cold":"Paracetamol",
        "Asthma":"Salbutamol",
        "Diabetes":"Metformin",
        "Hypertension":"Amlodipine",
        "Pneumonia":"Azithromycin"
    }

    disease = st.selectbox(
        "Select Disease",
        list(medicine_dict.keys())
    )

    st.success(
        f"Recommended Medicine: {medicine_dict[disease]}"
    )

   
with tab4:

    st.subheader("🔍 Disease Explorer")

    selected_disease = st.selectbox(
        "Select Disease",
        sorted(df["disease"].unique())
    )

    disease_df = filtered_df[
        filtered_df["disease"] == selected_disease
    ]

    st.metric(
        "Patients Found",
        len(disease_df)
    )

    st.dataframe(
        disease_df,
        use_container_width=True
    )


with tab5:

    st.subheader("🧬 Health Score Calculator")

    age = st.slider(
    "Age",
    1,
    100,
    30,
    key="health_age"
)

    bp = st.slider(
        "Blood Pressure Score",
        1,
        3,
        1
    )

    chol = st.slider(
        "Cholesterol Score",
        1,
        3,
        1
    )

    score = 100

    score -= age * 0.3
    score -= bp * 10
    score -= chol * 10

    score = max(score,0)

    st.metric(
        "Health Score",
        round(score)
    )

    if score >= 80:
        st.success("🟢 Excellent Health")
    elif score >= 60:
        st.warning("🟡 Moderate Health")
    else:
        st.error("🔴 High Health Risk")

# =====================================
# DATASET TAB
# =====================================

with tab6:

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    csv = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Data",
        csv,
        "healthcare_data.csv",
        "text/csv"
    )

# =====================================
# PROJECT SUMMARY
# =====================================

st.markdown("---")

st.success("""
🏥 Healthcare Intelligence Dashboard

Key Features:

• Disease Analytics
• Risk Assessment
• Medicine Recommendation
• Population Health Insights
• Interactive Filtering
• Downloadable Reports

This system demonstrates how AI and Data Analytics
can support healthcare decision-making and patient monitoring.
""")