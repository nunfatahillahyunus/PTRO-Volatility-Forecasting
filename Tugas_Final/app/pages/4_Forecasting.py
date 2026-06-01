#LIBRARY
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

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


st.title(
    "Volatility Forecasting"
)


#####################################################################
# Download PTRO data

ptro = yf.download(
    'PTRO.JK',
    start='2015-01-01',
    auto_adjust=True
)

ptro.columns = ptro.columns.get_level_values(0)

log_return = np.log(
    ptro['Close'] /
    ptro['Close'].shift(1)
)

log_return = log_return.dropna()

volatility = (
    log_return
    .rolling(window=21)
    .std()
)

volatility = volatility.dropna()

#####################################################################
# Plot

start_date = st.date_input( # setup tanggal slider
    "Start Date",
    volatility.index.min().date()
)

end_date = st.date_input(
    "End Date",
    volatility.index.max().date()
)

vol_filtered = volatility.loc[
    str(start_date):str(end_date)
]


fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=vol_filtered.index,
        y=vol_filtered.values,

        mode='lines',

        name='Volatility',

        line=dict(
            color='#00E5FF',
            width=2
        )

    )

)

fig.update_layout(

    title=f'PTRO Volatility ({start_date} to {end_date})',

    template='plotly_dark',

    height=600,

    xaxis_title='Date',

    yaxis_title='Volatility',

    hovermode='x unified',

    paper_bgcolor='#0E1117',

    plot_bgcolor='#0E1117'

)

st.plotly_chart(
    fig,
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Volatility",
    f"{vol_filtered.mean():.4f}"
)

col2.metric(
    "Max Volatility",
    f"{vol_filtered.max():.4f}"
)

col3.metric(
    "Min Volatility",
    f"{vol_filtered.min():.4f}"
)



#####################################################################
# Density grafik Volatilitas

density = gaussian_kde(
    volatility,
    bw_method=0.5
)

x = np.linspace(
    volatility.min(),
    volatility.max(),
    1000
)

y = density(x)

# Persentil
percentiles = [10, 25, 50, 75, 95]

percentile_values = np.percentile(
    volatility,
    percentiles
)

fig = go.Figure()

# KDE Curve

fig.add_trace(

    go.Scatter(

        x=x,

        y=y,

        mode='lines',

        fill='tozeroy',

        fillcolor='rgba(0,229,255,0.15)',

        line=dict(
            color='#00E5FF',
            width=4
        ),

        name='Density Curve'

    )

)

for p, x_val in zip(
    percentiles,
    percentile_values
):

    y_val = density(x_val)[0]

    # Vertical Line

    fig.add_shape(

        type='line',

        x0=x_val,
        x1=x_val,

        y0=0,
        y1=y_val,

        line=dict(
            color='#FF9800',
            width=2,
            dash='dash'
        )

    )

    # Marker

    fig.add_trace(

        go.Scatter(

            x=[x_val],

            y=[y_val],

            mode='markers+text',

            text=[f'{p}%'],

            textposition='top center',

            marker=dict(
                size=8,
                color='#FF9800'
            ),

            showlegend=False

        )

    )


fig.update_layout(

    title='Distribution of 21-Day Rolling Volatility',

    template='plotly_dark',

    paper_bgcolor='#0E1117',

    plot_bgcolor='#0E1117',

    xaxis_title='Volatility',

    yaxis_title='Density',

    height=600,

    showlegend=False

)

st.plotly_chart(
    fig,
    use_container_width=True
)






################################################################
#Treshold Volatility

p25 = percentile_values[1]
p75 = percentile_values[3]
p95 = percentile_values[4]

st.subheader(
    "Volatility Thresholds"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "P25",
    f"{p25:.4f}"
)

col2.metric(
    "P50",
    f"{percentile_values[2]:.4f}"
)

col3.metric(
    "P75",
    f"{p75:.4f}"
)

col4.metric(
    "P95",
    f"{p95:.4f}"
)

st.subheader(
    "Risk Classification"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success(
        f"Low Risk\n\n< {p25:.4f}"
    )

with col2:
    st.info(
        f"Medium Risk\n\n{p25:.4f} - {p75:.4f}"
    )

with col3:
    st.warning(
        f"High Risk\n\n{p75:.4f} - {p95:.4f}"
    )

with col4:
    st.error(
        f"Extreme Risk\n\n≥ {p95:.4f}"
    )



####################################################################
# VOLATILITY FORECASTING

st.subheader(
    "Volatility Forecasting"
)

# Model
model = load_model(
    BASE DIR / 'PTRO_LSTM_Volatility.h5',
    compile=False
)

# Scaler
scaler_X = joblib.load(
    BASE DIR / 'PTRO_scaler_X.pkl'
)

scaler_Y = joblib.load(
    BASE DIR / 'PTRO_scaler_Y.pkl'
)

#Variabel
log_return = np.log(
    ptro['Close'] /
    ptro['Close'].shift(1)
)

std_21 = (
    log_return
    .rolling(window=21)
    .std()
)

forecast_data = pd.DataFrame({

    'X_lr': log_return,
    'X_stdt': std_21

})

forecast_data = forecast_data.dropna()

# Scaling
X_scaled = scaler_X.transform(
    forecast_data[['X_lr', 'X_stdt']]
)


TIME_STEPS = 21

X_input = X_scaled[
    -TIME_STEPS:
]

X_input = X_input.reshape(
    1,
    TIME_STEPS,
    2
)

#Prediksi
pred_scaled = model.predict(
    X_input,
    verbose=0
)

pred_volatility = scaler_Y.inverse_transform(
    pred_scaled
)[0][0]

# Vol sekarang
latest_volatility = forecast_data[
    'X_stdt'
].iloc[-1]

change_pct = (

    (
        pred_volatility
        -
        latest_volatility
    )

    /

    latest_volatility

) * 100

#Klasifikasi risk
if pred_volatility < p25:

    risk_level = "Low Risk"

elif pred_volatility < p75:

    risk_level = "Medium Risk"

elif pred_volatility < p95:

    risk_level = "High Risk"

else:

    risk_level = "Extreme Risk"


# 3 Poin
col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Latest Volatility",
        f"{latest_volatility:.4f}"
    )

with col2:

    st.metric(
        "Predicted Volatility (t+1)",
        f"{pred_volatility:.4f}",
        f"{change_pct:.2f}%"
    )

with col3:

    st.metric(
        "Risk Level",
        risk_level
    )

# Risk Alert
if risk_level == "Low Risk":

    st.success(
        f"""
        LOW RISK

        Predicted volatility is
        {pred_volatility:.4f}.

        Market volatility is currently
        in a relatively stable range.
        """
    )

elif risk_level == "Medium Risk":

    st.info(
        f"""
        MEDIUM RISK

        Predicted volatility is
        {pred_volatility:.4f}.

        Volatility is within
        the normal historical range.
        """
    )

elif risk_level == "High Risk":

    st.warning(
        f"""
        HIGH RISK

        Predicted volatility is
        {pred_volatility:.4f}.

        Market uncertainty is increasing.
        """
    )

else:

    st.error(
        f"""
        EXTREME RISK

        Predicted volatility is
        {pred_volatility:.4f}.

        Volatility is extremely high. Trade With cautious"""
    )





# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Kelompok 4 | PTRO Volatility Forecasting using LSTM"
)
