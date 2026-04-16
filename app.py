import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm import weights as w
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
SAW = WSM
from pymcdm.helpers import rrankdata
from pymcdm import visuals

st.set_page_config(page_title="MCDM Dashboard", layout="wide")

# -----------------------------------------------------------------------------------------
# NEW: CUSTOM CSS STYLING
# -----------------------------------------------------------------------------------------
st.markdown("""
<style>
/* Make the Expander header text larger */
.streamlit-expanderHeader {
    font-size: 20px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------------------------------


st.title("Multi-Criteria Decision Making (MCDM) Dashboard")

# --- 1. DATA INPUT ---
st.sidebar.header("1. Upload or Edit Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Default data fallback
    data = {
        'alternative': ['A1', 'A2', 'A3'],
        'discharge': [2.5, 3.0, 4.0],
        'cost': [50, 60, 80],
        'wetlands': [0.9, 0.6, 0.1],
        'forest': [0.1, 0.6, 0.3],
        'social acceptance': [0.17, 0.83, 0.50]
    }   
    df = pd.DataFrame(data)

st.subheader("Decision Matrix")
st.markdown("Edit the matrix directly below or upload a new CSV file from the sidebar.")
# Using st.data_editor allows the user to edit the matrix dynamically
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Extract alternatives and criteria
alts_names = edited_df.iloc[:, 0].tolist()
criteria_names = edited_df.columns[1:]
alts_data = edited_df.iloc[:, 1:].to_numpy()

# -----------------------------------------------------------------------------------------
# NEW WIDGET 3: Expander with Side-by-Side Layout (Data Exploration)
# -----------------------------------------------------------------------------------------
with st.expander("📊 Explore Raw Data Distribution", expanded=True):
    st.markdown("Analyze how the alternatives compare across specific criteria before weighting.")
    
    col_controls, col_chart = st.columns([1, 2]) 
    
    with col_controls:
        selected_criterion = st.selectbox("Select Criterion:", criteria_names)
        max_val = edited_df[selected_criterion].max()
        min_val = edited_df[selected_criterion].min()
        st.info(f"**Highest Value:** {max_val}\n\n**Lowest Value:** {min_val}")

    with col_chart:
        st.bar_chart(
            data=edited_df, 
            x='alternative', 
            y=selected_criterion, 
            color='alternative', 
            use_container_width=True
        )
# -----------------------------------------------------------------------------------------

# --- 2. WEIGHTS & TYPES CONFIGURATION ---
st.sidebar.header("2. Criteria Configuration")
st.sidebar.markdown("Set weights and types for each criterion.")

weights_list = []
types_list = []

for col in criteria_names:
    st.sidebar.markdown(f"**{col}**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        # Slider for weights
        weight = st.slider(f"Weight", min_value=0.0, max_value=1.0, value=1.0/len(criteria_names), key=f"w_{col}")
        weights_list.append(weight)
    with c2:
        # Cost or Benefit option button
        ctype = st.radio("Type", options=["Benefit", "Cost"], key=f"t_{col}")
        types_list.append(1 if ctype == "Benefit" else -1)

# Normalize weights so they sum to 1
weights = np.array(weights_list)
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)
types = np.array(types_list)

# --- 3. METHOD SELECTION ---
st.sidebar.header("3. Select MCDM Methods")
available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW(),
    'MABAC': MABAC(),
    'ARAS': ARAS(),
    'WSM': WSM()
}

selected_method_names = st.sidebar.multiselect(
    "Choose evaluation methods:",
    list(available_methods.keys()),
    default=['TOPSIS', 'SAW']
)


# -----------------------------------------------------------------------------------------
# THE FIX: Initialize session state to remember if the analysis has been run
# -----------------------------------------------------------------------------------------
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# Clicking the button changes the memory state to True
if st.button("Run MCDM Analysis"):
    if not selected_method_names:
        st.warning("Please select at least one method from the sidebar.")
    else:
        st.session_state.analysis_run = True

# Now, we use the memory state to display the results, so they survive reruns!
if st.session_state.analysis_run and selected_method_names:
# -----------------------------------------------------------------------------------------

    methods = [available_methods[name] for name in selected_method_names]
    prefs = []
    ranks = []
    
    # Determine preferences and ranking for alternatives
    for method in methods:
        pref = method(alts_data, weights, types)
        rank = rrankdata(pref)
        
        prefs.append(pref)
        ranks.append(rank)
        
    pref_df = pd.DataFrame(zip(*prefs), columns=selected_method_names, index=alts_names).round(3)
    rank_df = pd.DataFrame(zip(*ranks), columns=selected_method_names, index=alts_names).astype(int)

    # -----------------------------------------------------------------------------------------
    # NEW WIDGET 1: Toggle Switch (Strict Interactive Input Widget)
    # -----------------------------------------------------------------------------------------
    first_method = selected_method_names[0]
    top_alt = rank_df[rank_df[first_method] == 1].index[0]
    
    st.markdown("The Winning Alternative")
    # This is a true input widget that changes the state of the app
    highlight_winner = st.toggle(f" Highlight the #1 Alternative ({top_alt}) in the tables below", value=True)
    
    # Function to apply Pandas styling to the dataframe rows based on the toggle
    def highlight_top_row(row):
        if highlight_winner and row.name == top_alt:
            return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
        return [''] * len(row)
    # -----------------------------------------------------------------------------------------

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preference Table")
        # We apply the highlight function to the dataframe before rendering
        st.dataframe(pref_df.style.apply(highlight_top_row, axis=1), use_container_width=True)
        
    with col2:
        st.subheader("Ranking Table")
        # We apply the highlight function to the dataframe before rendering
        st.dataframe(rank_df.style.apply(highlight_top_row, axis=1), use_container_width=True)
        
        # -----------------------------------------------------------------------------------------
        # NEW WIDGET 2: Download Button (Full-width styled block)
        # -----------------------------------------------------------------------------------------
        csv_data = rank_df.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Final Rankings as CSV",
            data=csv_data,
            file_name='mcdm_final_rankings.csv',
            mime='text/csv',
            type="primary",
            use_container_width=True,
            help="Export this table to Excel or CSV for reporting."
        )
        # -----------------------------------------------------------------------------------------

    # Plotting the polar chart
    st.subheader("Polar Ranking Plot")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150, tight_layout=True, subplot_kw=dict(projection='polar'))
    visuals.polar_plot(ranks, labels=selected_method_names, legend_ncol=2, ax=ax)
    st.pyplot(fig)
