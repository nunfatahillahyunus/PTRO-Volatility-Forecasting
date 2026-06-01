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

#####################################################################
st.subheader(
    "LSTM Overview"
)

overview_table = pd.DataFrame({

    'Component': [

        'Input Feature 1',
        'Input Feature 2',
        'Target Variable',
        'Time Steps',
        'Model Type',
        'Prediction Objective'

    ],

    'Description': [

        'X_lr (Log Return)',
        'X_stdt (Volatility t)',
        'Y (Volatility t+1)',
        '21 Days',
        'Long Short-Term Memory (LSTM)',
        'Predict Next-Day Volatility'

    ]

})

st.dataframe(
    overview_table,
    use_container_width=True
)

#####################################################################
# Model Architecture
st.subheader(
    "Model Architecture"
)

st.write(
    """
    Model menggunakan arsitektur
    Long Short-Term Memory (LSTM)
    untuk mempelajari pola temporal
    pada data volatilitas saham.
    """
)

architecture_table = pd.DataFrame({

    'Layer': [

        'LSTM Layer',
        'Dropout Layer',
        'Dense Layer',
        'Optimizer',
        'Loss Function',
        'Batch Size',
        'Epoch',
        'Early Stopping'

    ],

    'Configuration': [

        '128 Units',
        '0.2',
        '1 Output Neuron',
        'Adam',
        'Mean Squared Error (MSE)',
        '32',
        '100',
        'Patience = 5'

    ],

    'Function': [

        'Mempelajari pola time series',
        'Mengurangi overfitting',
        'Menghasilkan prediksi volatilitas',
        'Optimasi parameter model',
        'Mengukur error prediksi',
        'Jumlah data per training step',
        'Jumlah iterasi training',
        'Menghentikan training otomatis'

    ]

})

st.dataframe(
    architecture_table,
    use_container_width=True
)

#####################################################################
# Model Flow

st.subheader(
    "Model Flow"
)

st.markdown(
    """
<style>

.flow-box {
    background-color: #1A1D29;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 16px;
    border: 1px solid #2E3440;
    min-height: 90px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.arrow {
    text-align: center;
    color: white;
    font-size: 35px;
    margin-top: 20px;
    font-weight: bold;
}

</style>
""",
    unsafe_allow_html=True
)

col1, arrow1, col2, arrow2, col3, arrow3, col4, arrow4, col5 = st.columns(
    [2,0.5,2,0.5,2,0.5,2,0.5,2]
)

with col1: # box 1
    st.markdown(
        """
        <div class="flow-box">
        Sequence<br>(21 Hari)
        </div>
        """,
        unsafe_allow_html=True
    )

# ARROW
with arrow1:
    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )

with col2: # box 2
    st.markdown( 
        """
        <div class="flow-box">
        LSTM Layer<br>128 Units
        </div>
        """,
        unsafe_allow_html=True
    )

# ARROW
with arrow2:
    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="flow-box">
        Dropout<br>0.2
        </div>
        """,
        unsafe_allow_html=True
    )

# ARROW
with arrow3:
    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="flow-box">
        Dense Layer<br>1 Neuron
        </div>
        """,
        unsafe_allow_html=True
    )

# ARROW
with arrow4:
    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        """
        <div class="flow-box">
        Predicted<br>Volatility
        </div>
        """,
        unsafe_allow_html=True
    )

#####################################################################
# Plot train and validation loss
st.subheader(
    "Training vs Validation Loss"
)

st.image(
    BASE_DIR / 'Stat/training_validation_loss.png',
    use_container_width=True
)

#####################################################################
# Eval Metrics
st.subheader(
    "Evaluation Metrics"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="MAE",
        value="0.0496"
    )

with col2:
    st.metric(
        label="RMSE",
        value="0.0715"
    )



#####################################################################
# Plot Actual vs Predicted Volatility

st.subheader(
    "Actual vs Predicted Volatility"
)

st.image(
    BASE_DIR / 'Stat\actual_vs_predicted.png',
    use_container_width=True
)




# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Kelompok 4 | PTRO Volatility Forecasting using LSTM"
)
