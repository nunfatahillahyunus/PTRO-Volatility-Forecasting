#LIBRARY
import streamlit as st
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Logo vinix dan ptro
st.logo(
    BASE_DIR / 'logo.png'
)

st.sidebar.image(
    BASE_DIR / 'VINIX7.png',
    width=150
)

# Title
st.title(
    "Exploratory Data Analysis"
)

st.subheader(
    "Research Variables"
)

#####################################################################
# Tabel variabel penelitian
variable_table = pd.DataFrame({

    'Variable': [

        'X_lr',
        'X_stdt',
        'Y'

    ],

    'Description': [

        'Log return saham PTRO',
        'Rolling standard deviation (window = 21) pada waktu t',
        'Rolling standard deviation (window = 21) pada waktu t+1'

    ],

    'Role': [

        'Independent Variable',
        'Independent Variable',
        'Dependent Variable'

    ]

})

st.dataframe(
    variable_table,
    use_container_width=True
)

#####################################################################
# SPLIT TABLE

st.subheader(
    "Dataset Split Information"
)

split_table = pd.DataFrame({

    'Dataset': [
        'Training',
        'Validation',
        'Testing'
    ],

    'Percentage': [
        '80%',
        '10%',
        '10%'
    ],

    'Purpose': [

        'Digunakan untuk melatih model LSTM',
        'Digunakan untuk validasi model selama training',
        'Digunakan untuk evaluasi performa model'

    ]

})

st.dataframe(
    split_table,
    use_container_width=True
)

#####################################################################
# Scaler

st.subheader(
    "Min-Max Scaling"
)

st.write(
    """
    Penelitian ini menggunakan
    Min-Max Scaling untuk melakukan
    normalisasi data sebelum
    dimasukkan ke dalam model LSTM.
    """
)

st.latex(
    r'''
    X_{scaled} =
    \frac{X - X_{min}}
    {X_{max} - X_{min}}
    '''
)

st.write(
    """
    Min-Max Scaling digunakan untuk
    mengubah rentang data menjadi
    skala antara 0 dan 1. Proses scaling penting dalam
    model Long Short-Term Memory (LSTM)
    karena model neural network
    sangat sensitif terhadap skala data.

    Tanpa scaling, variabel dengan
    rentang nilai yang besar dapat
    mendominasi proses training
    dan menyebabkan model sulit
    melakukan pembelajaran secara optimal.
    """
)

#####################################################################
# Descriptive Statistics

data_train_scaled = pd.read_csv(
    BASE_DIR / 'CSV/data_train_scaled.csv'
)

data_val_scaled = pd.read_csv(
    BASE_DIR / 'CSV/data_val_scaled.csv'
)

data_test_scaled = pd.read_csv(
    BASE_DIR / 'CSV/data_test_scaled.csv'
)

st.subheader(
    "Descriptive Statistics"
)

# Function statistik
def descriptive_stats(data):

    stats = pd.DataFrame({

        'Mean': data.mean(),
        'Std': data.std(),
        'Median': data.median()

    })

    return stats

st.write("#### Training Data (2204)")

train_stats = descriptive_stats(
    data_train_scaled
)

st.dataframe(
    train_stats,
    use_container_width=True
)

st.write("#### Validation Data (257)")

val_stats = descriptive_stats(
    data_val_scaled
)

st.dataframe(
    val_stats,
    use_container_width=True
)

st.write("#### Testing Data (258)")

test_stats = descriptive_stats(
    data_test_scaled
)

st.dataframe(
    test_stats,
    use_container_width=True
)

#####################################################################
# Plot ts X dan Y

st.subheader(
    "Time Series Plot"
)

st.image(
    BASE_DIR / 'Stat/time_series_plot.png',
    use_container_width=True
)




# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Kelompok 4 | PTRO Volatility Forecasting using LSTM"
)
