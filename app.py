import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px

css_path = Path(__file__).with_name("bad2drivers.css")
with open(css_path) as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Bad Drivers Dashboard", page_icon="🚗", layout="wide")
st.title("🚦 Análisis de Conductores en Colisiones Fatales")
st.caption("Filtros: Alcohol-Impaired & Not Distracted")

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def find_col(df, target):
    t = norm(target)
    for c in df.columns:
        if norm(c) == t:
            return c
    kws = t.split()
    for c in df.columns:
        if all(k in norm(c) for k in kws):
            return c
    raise KeyError(f"Column not found: {target}")

def to_num_pct(s):
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    )

@st.cache_data
def load_data():
    df = pd.read_csv("bad-drivers.csv")
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    col_alc = find_col(df, "Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired")
    col_notd = find_col(df, "Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted")
    df[col_alc] = to_num_pct(df[col_alc])
    df[col_notd] = to_num_pct(df[col_notd])
    return df, col_alc, col_notd

df, COL_ALC, COL_NOTD = load_data()
COL_PREMIUM = find_col(df, "Car Insurance Premiums ($)")
df[COL_PREMIUM] = pd.to_numeric(df[COL_PREMIUM], errors="coerce")
COL_DRIVERS = find_col(df, "Number of drivers involved in fatal collisions per billion miles")

df["Premium_Group"] = pd.qcut(
    df[COL_PREMIUM],
    q=3,
    labels=["Low Premium", "Medium Premium", "High Premium"]
)

PALETTE = {
    "Low Premium": "#90CAF9",
    "Medium Premium": "#42A5F5",
    "High Premium": "#0D47A1",
}

st.sidebar.header("Filtros")

if "State" in df.columns:
    states = st.sidebar.multiselect(
        "Select State(s):",
        df["State"].unique(),
        default=list(df["State"].unique())
    )
else:
    states = None

alc_range = st.sidebar.slider(
    "Alcohol-Impaired (%)",
    float(df[COL_ALC].min()),
    float(df[COL_ALC].max()),
    (float(df[COL_ALC].min()), float(df[COL_ALC].max()))
)
notd_range = st.sidebar.slider(
    "Not Distracted (%)",
    float(df[COL_NOTD].min()),
    float(df[COL_NOTD].max()),
    (float(df[COL_NOTD].min()), float(df[COL_NOTD].max()))
)

premium_groups = st.sidebar.multiselect(
    "Insurance Premium Group:",
    options=list(df["Premium_Group"].cat.categories),
    default=list(df["Premium_Group"].cat.categories)
)

mask = (
    df[COL_ALC].between(alc_range[0], alc_range[1]) &
    df[COL_NOTD].between(notd_range[0], notd_range[1])
)
if states:
    mask &= df["State"].isin(states)
if premium_groups:
    mask &= df["Premium_Group"].isin(premium_groups)

filtered = df[mask]

c1, c2, c3 = st.columns(3)
c1.metric("Promedio Alcohol-Impaired", f"{filtered[COL_ALC].mean():.2f}%")
c2.metric("Promedio Not Distracted", f"{filtered[COL_NOTD].mean():.2f}%")
c3.metric("Registros filtrados", len(filtered))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Mapa por Estado",
    "Distribución Not Distracted",
    "Relación Alcohol vs Not Distracted",
    "Conductores por Estado y Prima",
    "Distribución Alcohol-Impaired",
])

sns.set_theme(style="whitegrid")

with tab1:
    st.subheader("Mapa de conductores por estado (fatal collisions per billion miles)")
    STATE_ABBREVS = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
        'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
        'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    map_data = filtered[["State", COL_DRIVERS]].copy()
    map_data["state_code"] = map_data["State"].map(STATE_ABBREVS)
    map_data = map_data.dropna(subset=["state_code"])
    if not map_data.empty:
        fig_map = px.choropleth(
            map_data,
            locations="state_code",
            locationmode="USA-states",
            color=COL_DRIVERS,
            hover_name="State",
            scope="usa",
            color_continuous_scale="Blues",
            labels={COL_DRIVERS: "Drivers per billion miles"}
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#0D47A1",
            plot_bgcolor="#0D47A1",
            font_color="white"
        )
        fig_map.update_geos(bgcolor="#0D47A1")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No hay datos para mostrar en el mapa con los filtros seleccionados.")

with tab5:
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered[COL_ALC], bins=15, ax=ax1, color="#42A5F5")
    ax1.set_title("Distribution: Alcohol-Impaired (%)")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered[COL_NOTD], bins=15, ax=ax2, color="#66BB6A")
    ax2.set_title("Distribution: Not Distracted (%)")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=filtered, x=COL_ALC, y=COL_NOTD, ax=ax3, color="#AB47BC")
    ax3.set_title("Relation: Alcohol-Impaired vs Not Distracted")
    st.pyplot(fig3)

with tab4:
    st.subheader("Conductores en colisiones fatales por Estado y grupo de prima de seguro")
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    plot_data = filtered.sort_values("State")
    sns.barplot(
        data=plot_data,
        x="State",
        y=COL_DRIVERS,
        hue="Premium_Group",
        palette=PALETTE,
        ax=ax4
    )
    ax4.set_title(
        "Drivers involved in fatal collisions per billion miles\npor Estado y grupo de Car Insurance Premium"
    )
    ax4.set_xlabel("State")
    ax4.set_ylabel("Drivers per billion miles")
    ax4.tick_params(axis="x", rotation=90)
    for bar in ax4.patches:
        height = bar.get_height()
        r, g, b, _ = bar.get_facecolor()
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = 'white' if brightness < 0.5 else 'black'
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha='center',
            va='bottom',
            color=text_color,
            fontsize=8
        )
    st.pyplot(fig4)


