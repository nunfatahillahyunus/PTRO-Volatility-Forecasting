# =====================================================
# IMPORT LIBRARY
# =====================================================

import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PTRO Volatility Forecasting",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# LOGO
# =====================================================

st.logo(
    'logo.png'
)

st.sidebar.image(
    'VINIX7.png',
    width=150
)

# =====================================================
# HERO SECTION
# =====================================================

st.title(
    "PTRO Volatility Forecasting using LSTM"
)

st.markdown(
    """
    ### Web Application for Volatility Forecasting and Risk Analysis

    Sistem ini dikembangkan untuk menganalisis dan memprediksi
    volatilitas saham PT Petrosea Tbk (PTRO) menggunakan
    model Long Short-Term Memory (LSTM).
    """
)

st.markdown("---")

# =====================================================
# ABOUT PTRO
# =====================================================

st.header(
    "About PTRO"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Kode Saham",
    "PTRO.JK"
)

col2.metric(
    "Sektor",
    "Energy"
)

col3.metric(
    "Subsektor",
    "Minyak, Gas, dan Batubara"
)

col4.metric(
    "Tanggal Pencatatan",
    "1990-05-21"
)

st.write(
    """
    PT Petrosea Tbk (PTRO) merupakan perusahaan yang bergerak
    di bidang jasa pertambangan, rekayasa, konstruksi, dan logistik.
    Saham PTRO termasuk salah satu saham sektor pertambangan yang
    memiliki karakteristik volatilitas yang cukup tinggi sehingga
    menarik untuk dianalisis dari perspektif risiko investasi.
    """
)

# =====================================================
# PROJECT OBJECTIVE
# =====================================================

st.header(
    "Project Objective"
)

st.write(
    """
    Tujuan proyek ini adalah membangun model Long Short-Term Memory (LSTM)
    untuk memprediksi volatilitas saham PTRO berdasarkan data historis
    sehingga dapat membantu investor memahami tingkat risiko pasar
    secara lebih terukur.
    """
)

# =====================================================
# ANALYSIS WORKFLOW
# =====================================================

st.header(
    "Analysis Workflow"
)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

c1.success("Data\nCollection")
c2.info("EDA")
c3.info("Feature\nEngineering")
c4.info("Data\nSplitting")
c5.info("LSTM\nModeling")
c6.info("Evaluation")
c7.warning("Forecasting")

st.markdown(
    """
    Data historis PTRO diperoleh dari Yahoo Finance,
    kemudian dilakukan Exploratory Data Analysis (EDA),
    feature engineering, training model LSTM,
    evaluasi performa model, dan forecasting volatilitas.
    """
)

# =====================================================
# DATASET OVERVIEW
# =====================================================

st.header(
    "Dataset Overview"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Period",
    "2015 - Present"
)

col2.metric(
    "Variables",
    "Close"
)

col3.metric(
    "Features",
    "Log Return and Volatility t"
)

col4.metric(
    "Target",
    "Volatility t+1"
)

# =====================================================
# APPLICATION FEATURES
# =====================================================

st.header(
    "Application Features"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    with st.container(border=True):

        st.subheader("Stock Data")

        st.write(
        """
        Historical stock prices,
        volume, and price visualization.
        """
        )

with col2:

    with st.container(border=True):

        st.subheader("EDA")

        st.write(
        """
        Return analysis,
        volatility analysis,
        and risk classification.
        """
        )

with col3:

    with st.container(border=True):

        st.subheader("LSTM")

        st.write(
        """
        Model architecture,
        training process,
        and evaluation metrics.
        """
        )

with col4:

    with st.container(border=True):

        st.subheader("Forecasting")

        st.write(
        """
        Volatility prediction
        and risk assessment.
        """
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Kelompok 4 | PTRO Volatility Forecasting using LSTM"
)