#LIBRARY
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

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
##

st.title(
    "PTRO Stock Analysis"
)

st.write(
    "Aplikasi prediksi volatilitas saham PTRO menggunakan model LSTM."
)

# DOWNLOAD DATA
ptro = yf.download(
    'PTRO.JK',
    start='2015-01-01',
    progress=False
)

ptro.index = pd.to_datetime(ptro.index)

if isinstance(ptro.columns, pd.MultiIndex):
    ptro.columns = ptro.columns.get_level_values(0)


ptro = ptro[
    ptro['Volume'] > 0
]

st.subheader("Stock Data")

st.dataframe(
    ptro.tail()
)

#####################################################################
# Price Chart

ptro_2024 = ptro.loc[
    ptro.index >= '2025-01-01'
]

ptro_2024['MA21'] = (
    ptro_2024['Close']
    .rolling(window=21)
    .mean()
)

st.subheader(
    "Price Chart"
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

fig.patch.set_facecolor('#0E1117')

ax.set_facecolor('#0E1117')

ax.plot(
    ptro_2024.index,
    ptro_2024['Close'],
    color='#00E5FF',
    linewidth=1.5,
    label='Close Price'
)

ax.plot(
    ptro_2024.index,
    ptro_2024['MA21'],
    color='#FF9800',
    linewidth=2,
    label='MA21'
)

ax.set_title(
    'Closing Price and MA21',
    color='white',
    fontsize=14
)

ax.set_xlabel(
    'Date',
    color='white'
)

ax.set_ylabel(
    'Price',
    color='white'
)

ax.tick_params(
    colors='white'
)

ax.grid(
    alpha=0.2
)

for spine in ax.spines.values():
    spine.set_color('white')

ax.legend(
    facecolor='#0E1117',
    edgecolor='white',
    labelcolor='white'
)

st.pyplot(fig)

#####################################################################
# Plot Log return

log_return = np.log(
    ptro['Close'] /
    ptro['Close'].shift(1)
)

log_return = log_return.dropna()

st.subheader(
    "Log Return"
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

fig.patch.set_facecolor('#0E1117')

ax.set_facecolor('#0E1117')

ax.plot(
    log_return.index,
    log_return,
    color='#00E5FF',
    linewidth=1
)

ax.set_title(
    'Log Return',
    color='white',
    fontsize=14
)

ax.set_xlabel(
    'Date',
    color='white'
)

ax.set_ylabel(
    'Log Return',
    color='white'
)

ax.tick_params(
    colors='white'
)

ax.grid(
    alpha=0.2
)

for spine in ax.spines.values():
    spine.set_color('white')

st.pyplot(fig)

#####################################################################
# Log return Distribution

st.subheader(
    "Log Return Distribution"
)

fig, ax = plt.subplots(
    figsize=(10,5)
)

# Background
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')

# Histogram + KDE
sns.histplot(
    log_return,
    bins=50,
    stat='density',
    kde=True,
    color='#00E5FF',
    edgecolor='white',
    alpha=0.6,
    ax=ax
)

# Title
ax.set_title(
    'Log Return Distribution',
    fontsize=14,
    color='white'
)

# Labels
ax.set_xlabel(
    'Log Return',
    color='white'
)

ax.set_ylabel(
    'Density',
    color='white'
)

# Tick color
ax.tick_params(
    colors='white'
)

# Grid
ax.grid(
    alpha=0.2,
    color='white'
)

# Spine color
for spine in ax.spines.values():
    spine.set_color('white')

st.pyplot(fig)


#####################################################################
#Plot STDEV 21 Hari

std_21 = log_return.rolling(
    window=21
).std()

std_21 = std_21.dropna()

st.subheader(
    "Standard Deviation (21 Days)"
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

fig.patch.set_facecolor('#0E1117')

ax.set_facecolor('#0E1117')

ax.plot(
    std_21.index,
    std_21,
    color='#00E5FF',
    linewidth=1.5,
    label='Rolling Std (21)'
)

ax.set_title(
    'Standard Deviation (21 Days)',
    color='white',
    fontsize=14
)

ax.set_xlabel(
    'Date',
    color='white'
)

ax.set_ylabel(
    'Standard Deviation',
    color='white'
)

ax.tick_params(
    colors='white'
)

ax.grid(
    alpha=0.2
)

for spine in ax.spines.values():
    spine.set_color('white')

ax.legend(
    facecolor='#0E1117',
    edgecolor='white',
    labelcolor='white'
)

st.pyplot(fig)




# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Kelompok 4 | PTRO Volatility Forecasting using LSTM"
)
