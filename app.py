"""
F1 Podium Predictor — Streamlit App
Deploy on Streamlit Cloud: share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #E10600 0%, #FF6B6B 50%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    margin-bottom: 0;
  }
  .subtitle {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
  }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
  }
  .metric-label {
    color: #888;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
  }
  .metric-value {
    color: #fff;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 0.2rem;
  }
  .podium-yes {
    background: linear-gradient(135deg, #0d4a1a 0%, #1a7a2e 100%);
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .podium-no {
    background: linear-gradient(135deg, #4a0d0d 0%, #7a1a1a 100%);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .result-text {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 3px;
  }
  .stButton > button {
    background: linear-gradient(135deg, #E10600, #FF4444);
    color: white;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 2px;
    border: none;
    border-radius: 8px;
    padding: 0.8rem 2rem;
    width: 100%;
    transition: all 0.3s ease;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(225, 6, 0, 0.4);
  }
  .section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    color: #E10600;
    font-weight: 700;
    letter-spacing: 2px;
    border-bottom: 1px solid #333;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }
  .info-box {
    background: #1a1a2e;
    border-left: 3px solid #E10600;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #ccc;
  }
  .team-badge {
    display: inline-block;
    background: #E10600;
    color: white;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)


# ── Model Training / Loading ──────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train a demo model if CSVs not found."""
    MODEL_PATH = "rf_model.pkl"
    META_PATH  = "model_meta.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta  = joblib.load(META_PATH)
        return model, meta, "loaded"

    # ── Try loading real F1 data ──────────────────────────────────
    try:
        results      = pd.read_csv('results.csv')
        races        = pd.read_csv('races.csv')[['raceId','year','circuitId']]
        constructors = pd.read_csv('constructors.csv')[['constructorId','name']].rename(columns={'name':'constructor_name'})
        drivers      = pd.read_csv('drivers.csv')[['driverId','driverRef']]
        circuits     = pd.read_csv('circuits.csv')[['circuitId','name']].rename(columns={'name':'circuit_name'})

        df = (results[['raceId','driverId','constructorId','grid','laps','positionOrder','points']]
              .merge(races, on='raceId')
              .merge(constructors, on='constructorId')
              .merge(drivers, on='driverId')
              .merge(circuits, on='circuitId'))

        df.replace('\\N', np.nan, inplace=True)
        for col in ['grid','laps','positionOrder','points']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['positionOrder','grid','laps'], inplace=True)
        df['podium'] = (df['positionOrder'] <= 3).astype(int)

        constructor_list = df['constructor_name'].dropna().unique().tolist()
        driver_list      = df['driverRef'].dropna().unique().tolist()
        circuit_list     = df['circuit_name'].dropna().unique().tolist()

        for col in ['driverId','constructorId','circuitId']:
            df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

        df['front_row']  = (df['grid'] <= 3).astype(int)
        max_laps         = df.groupby('raceId')['laps'].transform('max')
        df['laps_ratio'] = (df['laps'] / max_laps.replace(0, np.nan)).fillna(0)

        FEATURES = ['grid','year','front_row','laps_ratio',
                    'driverId_enc','constructorId_enc','circuitId_enc']

        X = df[FEATURES]
        y = df['podium']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        meta = {
            'constructors'      : sorted(constructor_list),
            'drivers'           : sorted(driver_list),
            'circuits'          : sorted(circuit_list),
            'constructor_map'   : dict(zip(
                df['constructor_name'].astype(str),
                df['constructorId_enc'].astype(int))),
            'driver_map'        : dict(zip(
                df['driverRef'].astype(str),
                df['driverId_enc'].astype(int))),
            'circuit_map'       : dict(zip(
                df['circuit_name'].astype(str),
                df['circuitId_enc'].astype(int))),
            'feature_names'     : FEATURES,
            'year_min'          : int(df['year'].min()),
            'year_max'          : int(df['year'].max()),
        }
        joblib.dump(model, MODEL_PATH)
        joblib.dump(meta, META_PATH)
        return model, meta, "trained_real"

    except FileNotFoundError:
        pass

    # ── Demo model with synthetic data ────────────────────────────
    np.random.seed(42)
    N = 5000
    grid      = np.random.randint(1, 21, N)
    year      = np.random.randint(1990, 2024, N)
    front_row = (grid <= 3).astype(int)
    laps_ratio= np.random.uniform(0.5, 1.0, N)
    drv_enc   = np.random.randint(0, 50, N)
    con_enc   = np.random.randint(0, 20, N)
    cir_enc   = np.random.randint(0, 30, N)

    prob_podium = np.clip(
        0.45 - 0.02*grid + 0.15*front_row + 0.05*laps_ratio + 
        np.random.normal(0, 0.05, N), 0, 1)
    podium = (np.random.random(N) < prob_podium).astype(int)

    X = pd.DataFrame({
        'grid': grid, 'year': year, 'front_row': front_row,
        'laps_ratio': laps_ratio, 'driverId_enc': drv_enc,
        'constructorId_enc': con_enc, 'circuitId_enc': cir_enc
    })
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42)
    model.fit(X, podium)

    CONSTRUCTORS = ["Red Bull Racing","Mercedes","Ferrari","McLaren",
                    "Aston Martin","Alpine","Williams","AlphaTauri",
                    "Alfa Romeo","Haas"]
    DRIVERS = ["Max Verstappen","Lewis Hamilton","Charles Leclerc",
               "Sergio Perez","Carlos Sainz","Lando Norris",
               "Fernando Alonso","George Russell","Oscar Piastri",
               "Lance Stroll"]
    CIRCUITS = ["Bahrain","Saudi Arabia","Australia","Azerbaijan",
                "Miami","Monaco","Spain","Canada","Austria",
                "Britain","Hungary","Belgium","Netherlands",
                "Italy","Singapore","Japan","Qatar","USA",
                "Mexico","Brazil","Abu Dhabi"]

    meta = {
        'constructors'    : CONSTRUCTORS,
        'drivers'         : DRIVERS,
        'circuits'        : CIRCUITS,
        'constructor_map' : {c: i for i, c in enumerate(CONSTRUCTORS)},
        'driver_map'      : {d: i for i, d in enumerate(DRIVERS)},
        'circuit_map'     : {c: i for i, c in enumerate(CIRCUITS)},
        'feature_names'   : list(X.columns),
        'year_min'        : 1990,
        'year_max'        : 2024,
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    return model, meta, "demo"


# ── Predict function ─────────────────────────────────────────────
def predict_podium(model, meta, grid, year, constructor, driver, circuit, laps_ratio=1.0):
    front_row     = 1 if grid <= 3 else 0
    con_enc       = meta['constructor_map'].get(constructor, 0)
    drv_enc       = meta['driver_map'].get(driver, 0)
    cir_enc       = meta['circuit_map'].get(circuit, 0)
    X             = pd.DataFrame([[grid, year, front_row, laps_ratio,
                                   drv_enc, con_enc, cir_enc]],
                                 columns=meta['feature_names'])
    prob          = model.predict_proba(X)[0][1]
    pred          = model.predict(X)[0]
    return pred, prob


# ── Load model ────────────────────────────────────────────────────
model, meta, source = load_or_train_model()


# ── Header ───────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<p class="main-title">🏎 F1 PODIUM PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Machine Learning · Random Forest · Formula 1 Analysis</p>', unsafe_allow_html=True)

if source == "demo":
    st.info("**Demo mode** — running on synthetic data. Add your Kaggle F1 CSVs (results.csv, races.csv, drivers.csv, constructors.csv, circuits.csv) to the project folder and restart to use real data.", icon="ℹ️")
elif source == "trained_real":
    st.success("Model trained on real F1 dataset!", icon="✅")

st.markdown("---")

# ── Main layout ───────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: Inputs ─────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-header">RACE PARAMETERS</p>', unsafe_allow_html=True)

    constructor = st.selectbox(
        "Constructor / Team",
        options=meta['constructors'],
        help="Select the F1 constructor"
    )
    driver = st.selectbox(
        "Driver",
        options=meta['drivers'],
        help="Select the driver"
    )
    circuit = st.selectbox(
        "Circuit",
        options=meta['circuits'],
        help="Select the race circuit"
    )

    col_g, col_y = st.columns(2)
    with col_g:
        grid_pos = st.number_input(
            "Grid Position",
            min_value=1, max_value=20, value=1,
            help="Starting grid position (1 = pole position)"
        )
    with col_y:
        year = st.number_input(
            "Season Year",
            min_value=meta['year_min'],
            max_value=meta['year_max'],
            value=2023
        )

    laps_ratio = st.slider(
        "Race Completion Ratio",
        min_value=0.0, max_value=1.0, value=1.0, step=0.01,
        help="1.0 = completed full race, < 1.0 = DNF/retired"
    )

    st.markdown("")
    predict_btn = st.button("PREDICT PODIUM FINISH", use_container_width=True)

    st.markdown('<p class="section-header">GRID INSIGHT</p>', unsafe_allow_html=True)
    front_row_flag = "Yes" if grid_pos <= 3 else "No"
    st.markdown(f"""
    <div class="info-box">
        <b>Starting position:</b> P{grid_pos}<br>
        <b>Front row:</b> {front_row_flag}<br>
        <b>Race completion:</b> {laps_ratio*100:.0f}%<br>
        <b>Era:</b> {"Modern (turbo hybrid)" if year >= 2014 else "V8 era" if year >= 2006 else "Classic"}
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: Prediction + Charts ────────────────────────────────────
with right_col:
    st.markdown('<p class="section-header">PREDICTION</p>', unsafe_allow_html=True)

    if predict_btn or True:
        pred, prob = predict_podium(
            model, meta, grid_pos, year,
            constructor, driver, circuit, laps_ratio)

        # ── Result card ───────────────────────────────────────────
        if pred == 1:
            st.markdown(f"""
            <div class="podium-yes">
              <div class="result-text" style="color:#2ecc71">PODIUM FINISH</div>
              <div style="color:#aaa;margin-top:0.5rem;font-size:0.9rem;letter-spacing:2px">
                PROBABILITY
              </div>
              <div style="color:#2ecc71;font-family:'Orbitron',monospace;
                          font-size:3rem;font-weight:900">
                {prob*100:.1f}%
              </div>
              <div style="color:#aaa;font-size:0.85rem;margin-top:0.5rem">
                {driver} is predicted to finish P1, P2, or P3
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="podium-no">
              <div class="result-text" style="color:#e74c3c">NO PODIUM</div>
              <div style="color:#aaa;margin-top:0.5rem;font-size:0.9rem;letter-spacing:2px">
                PROBABILITY
              </div>
              <div style="color:#e74c3c;font-family:'Orbitron',monospace;
                          font-size:3rem;font-weight:900">
                {prob*100:.1f}%
              </div>
              <div style="color:#aaa;font-size:0.85rem;margin-top:0.5rem">
                {driver} is predicted to finish outside the top 3
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Probability gauge ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 1.2))
        fig.patch.set_facecolor('#0e0e1a')
        ax.set_facecolor('#0e0e1a')
        ax.barh(0, 1, color='#1a1a2e', height=0.5)
        bar_color = '#2ecc71' if prob >= 0.5 else '#e74c3c'
        ax.barh(0, prob, color=bar_color, height=0.5, alpha=0.9)
        ax.axvline(0.5, color='#555', linewidth=1, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.text(0.02, 0, '0%', color='#555', va='center', fontsize=9)
        ax.text(0.5, 0, '50%', color='#555', va='center', ha='center', fontsize=9)
        ax.text(0.98, 0, '100%', color='#555', va='center', ha='right', fontsize=9)
        ax.text(prob, 0.28, f'{prob*100:.1f}%',
                color=bar_color, va='bottom', ha='center',
                fontsize=11, fontweight='bold')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Grid position sweep chart ─────────────────────────────
        st.markdown('<p class="section-header">GRID POSITION ANALYSIS</p>', unsafe_allow_html=True)

        grid_positions = list(range(1, 21))
        probs_by_grid  = []
        for g in grid_positions:
            _, p = predict_podium(model, meta, g, year,
                                  constructor, driver, circuit, laps_ratio)
            probs_by_grid.append(p * 100)

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0e0e1a')
        ax2.set_facecolor('#0e0e1a')

        colors_bar = ['#E10600' if g == grid_pos else
                      ('#2ecc71' if probs_by_grid[i] >= 50 else '#334')
                      for i, g in enumerate(grid_positions)]
        bars = ax2.bar(grid_positions, probs_by_grid,
                       color=colors_bar, edgecolor='none', width=0.7)
        ax2.axhline(50, color='#555', linewidth=1, linestyle='--', alpha=0.7)
        ax2.set_xlabel('Grid Position', color='#888', fontsize=9)
        ax2.set_ylabel('Podium Probability (%)', color='#888', fontsize=9)
        ax2.set_title(f'{constructor} — Podium % by Grid Position',
                      color='#ccc', fontsize=10, pad=8)
        ax2.tick_params(colors='#666', labelsize=8)
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0.3, 20.7)
        ax2.set_ylim(0, max(probs_by_grid) * 1.2 + 5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()


# ── Bottom: Feature importances + Model stats ─────────────────────
st.markdown("---")
st.markdown('<p class="section-header">MODEL INSIGHTS</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    importances = model.feature_importances_
    feat_names  = ['Grid Pos', 'Year', 'Front Row', 'Laps Ratio',
                   'Driver', 'Constructor', 'Circuit']
    sorted_idx  = np.argsort(importances)

    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    fig3.patch.set_facecolor('#0e0e1a')
    ax3.set_facecolor('#0e0e1a')
    colors_fi = ['#E10600' if i == sorted_idx[-1] else '#334'
                 for i in range(len(importances))]
    colors_fi_s = [colors_fi[i] for i in sorted_idx]
    ax3.barh([feat_names[i] for i in sorted_idx],
             importances[sorted_idx],
             color=colors_fi_s, edgecolor='none')
    ax3.set_title('Feature Importances', color='#ccc', fontsize=10, pad=8)
    ax3.tick_params(colors='#888', labelsize=8)
    ax3.spines['bottom'].set_color('#333')
    ax3.spines['left'].set_color('#333')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

with col2:
    st.markdown("**Model Configuration**")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Algorithm</div>
      <div style="color:#fff;font-size:1rem;font-weight:600;margin-top:4px">Random Forest</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Trees</div>
      <div class="metric-value">100</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Max Depth</div>
      <div class="metric-value">10</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("**How It Works**")
    st.markdown("""
    <div class="info-box">
      <b>1. Grid Position</b><br>
      Starting from pole gives ~40% podium probability vs ~2% from P10+
    </div>
    <div class="info-box">
      <b>2. Constructor</b><br>
      Dominant teams (Red Bull, Mercedes, Ferrari) significantly boost podium odds
    </div>
    <div class="info-box">
      <b>3. Race Completion</b><br>
      DNF (ratio &lt; 1.0) dramatically reduces predicted podium probability
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:0.8rem;padding:1rem 0">
  F1 Podium Predictor · Built with Streamlit & Scikit-learn · 
  Dataset: Ergast F1 World Championship (Kaggle) · ML Practical Assignment
</div>
""", unsafe_allow_html=True)
