"""Streamlit UI for the Bloei Rekenmodule."""

import os
# Forceer de Streamlit theme instellingen (belangrijk voor Azure Web App deployment)
os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] = "#0f494f"

import streamlit as st
from datetime import date, datetime
from bloei_rekenmodel import RekenInput, bereken_kosten, EenmaligeCashflow
import altair as alt
import pandas as pd

# Bloei huisstijlkleuren
BLOEI_PINK = "#ff787c"
BLOEI_PETROL = "#0f494f"
BLOEI_WARMGREY = "#eeeae9"
POSITIVE_COLOR = "#1db09e"
NEGATIVE_COLOR = "#b34025"

# Profile ordering (offensive -> defensive)
PROFIEL_ORDER = ["Zeer offensief", "Offensief", "Neutraal", "Matig defensief", "Defensief", "Niet beleggen"]

def allowed_profielen_for_horizon(horizon_jaren: int) -> list[str]:
    """Return allowed profile options for a given horizon, most-offensive -> most-defensive."""
    h = float(horizon_jaren)
    if h > 14:
        return PROFIEL_ORDER[:]
    if 10 <= h:
        return PROFIEL_ORDER[1:]
    if 8 <= h:
        return PROFIEL_ORDER[2:]
    if 6 <= h:
        return PROFIEL_ORDER[3:]
    if 3 < h:
        return PROFIEL_ORDER[4:]
    return ["Niet beleggen"]

def default_profiel_for_horizon(horizon_jaren: int) -> str:
    opts = allowed_profielen_for_horizon(horizon_jaren)
    return opts[0]

def fmt_nl_number(amount: float, decimals: int = 2) -> str:
    us = f"{float(amount):,.{decimals}f}"
    return us.replace(",", "_").replace(".", ",").replace("_", ".")

def fmt_pct_nl(value: float, decimals: int = 2) -> str:
    return f"{fmt_nl_number(value, decimals)}%"

def fmt_eur_nl(amount: float, decimals: int = 2) -> str:
    return "€ " + fmt_nl_number(amount, decimals)

def parse_nl_number(raw: str) -> float:
    cleaned = raw.strip().replace("€", "").replace("EUR", "").replace("eur", "").replace(" ", "")
    if cleaned == "":
        raise ValueError("Leeg bedrag")
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    return float(cleaned)

def _sync_currency_input(display_key: str, value_key: str, min_value: float, decimals: int) -> None:
    try:
        value = parse_nl_number(st.session_state[display_key])
        if value < min_value:
            raise ValueError("Onder minimum")
        st.session_state[value_key] = value
        st.session_state[display_key] = fmt_nl_number(value, decimals)
        st.session_state[f"{display_key}_error"] = ""
    except Exception:
        st.session_state[f"{display_key}_error"] = "Gebruik formaat zoals 100.000,00."

def currency_text_input(
    label: str,
    *,
    key: str,
    default: float,
    min_value: float = 0.0,
    decimals: int = 2,
) -> float:
    display_key = f"{key}_display"
    value_key = f"{key}_value"
    error_key = f"{display_key}_error"

    if value_key not in st.session_state:
        st.session_state[value_key] = float(default)
    if display_key not in st.session_state:
        st.session_state[display_key] = fmt_nl_number(default, decimals)
    if error_key not in st.session_state:
        st.session_state[error_key] = ""

    st.text_input(
        label,
        key=display_key,
        on_change=_sync_currency_input,
        args=(display_key, value_key, min_value, decimals),
    )

    if st.session_state[error_key]:
        st.error(st.session_state[error_key])

    return float(st.session_state[value_key])

def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)

if 'eenmalige_cashflows' not in st.session_state:
    st.session_state.eenmalige_cashflows = []

st.set_page_config(
    page_title="Bloei rekenmodule",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist CSS
st.markdown(
    f"""
<style>
:root {{
    --bloei-pink: {BLOEI_PINK};
    --bloei-petrol: {BLOEI_PETROL};
    --bloei-warmgrey: {BLOEI_WARMGREY};
    --bloei-positive: {POSITIVE_COLOR};
    --bloei-negative: {NEGATIVE_COLOR};
}}
</style>
""" + """
<style>
/* Reset button color to petrol globally, regardless of environment */
div.stButton > button:first-child {
    background-color: var(--bloei-petrol) !important;
    color: white !important;
    border-color: var(--bloei-petrol) !important;
}
div.stButton > button:first-child:hover {
    background-color: #0c393e !important;
    border-color: #0c393e !important;
    color: white !important;
}
div.stButton > button:first-child:active {
    background-color: #092e32 !important;
    border-color: #092e32 !important;
    color: white !important;
}

/* Reset slider color to petrol */
div.stSlider div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--bloei-petrol) !important;
    border-color: var(--bloei-petrol) !important;
    border-width: 2px !important;
    box-shadow: none !important;
}
div.stSlider div[data-baseweb="slider"] div[role="slider"]:hover,
div.stSlider div[data-baseweb="slider"] div[role="slider"]:focus {
    box-shadow: 0 0 0 0.2rem rgba(15, 73, 79, 0.2) !important;
}


/* Modern Metric Containers - fully transparent without beige background */
div[data-testid="metric-container"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
div[data-testid="metric-container"] label {
    font-weight: 500;
    font-size: 0.95rem;
    opacity: 0.8;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: var(--bloei-petrol) !important;
    font-weight: 700;
    font-size: 2.2rem;
}
.bloei-note {
    margin: 0.5rem 0 0;
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.7;
}
.bloei-positive {
    color: var(--bloei-positive);
    font-weight: 600;
}
.bloei-negative {
    color: var(--bloei-negative);
    font-weight: 600;
}
.kosten-open-table {
    width: 100%;
    margin: 1rem 0;
    border-collapse: collapse;
    color: var(--text-color);
}
.kosten-open-table th {
    text-align: left;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0.75rem 0.5rem;
    border-bottom: 2px solid rgba(128, 128, 128, 0.2);
}
.kosten-open-table td {
    padding: 0.75rem 0.5rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.1);
}
.kosten-open-table th:last-child,
.kosten-open-table td:last-child {
    text-align: right;
}
.bloei-title-petrol {
    color: var(--bloei-petrol) !important;
    margin: 0 0 0.5rem;
    font-weight: 600;
}
.highlight-row {
    background-color: rgba(15, 73, 79, 0.05);
}

/* BaseWeb Slider Global Override */
div[data-baseweb="slider"] {
    --primary-color: var(--bloei-petrol) !important;
}

/* 1. The label/value above the thumb */
div[data-testid="stThumbValue"],
div[data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--bloei-petrol) !important;
}

/* 2. The thumb itself (the draggable circle) */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--bloei-petrol) !important;
    border-color: var(--bloei-petrol) !important;
}

/* 3. The thumb's glow when hovered/focused */
div[data-baseweb="slider"] div[role="slider"]:focus,
div[data-baseweb="slider"] div[role="slider"]:hover {
    box-shadow: 0 0 0 0.2rem rgba(15, 73, 79, 0.2) !important;
}

/* 4. Active track segment */
div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
    background-color: var(--bloei-petrol) !important;
}

/* Optimalisaties voor Dark Mode */
@media (prefers-color-scheme: dark) {
    div[data-testid="metric-container"] label,
    .bloei-note {
        color: #e0e0e0 !important;
    }
    .kosten-open-table th {
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }
    .kosten-open-table td {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .highlight-row {
        background-color: rgba(15, 73, 79, 0.15) !important;
    }
    /* We ensure that the petrol color remains the original one (#0f494f) */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"],
    .bloei-title-petrol {
        color: var(--bloei-petrol) !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR: Input Parameters
# ==========================================
with st.sidebar:
    st.title("Parameters")
    
    startvermogen = currency_text_input(
        "Startvermogen (€)",
        key="startvermogen",
        default=100000.0,
        min_value=0.01,
    )
    
    startdatum = st.date_input("Startdatum", value=date.today())
    
    horizon_jaren = st.slider("Horizon (jaren)", min_value=0, max_value=40, value=10)
    end_date = _add_years(startdatum, int(horizon_jaren))
    
    profiel_opties = allowed_profielen_for_horizon(int(horizon_jaren))
    
    with st.expander("Profiel & simulatie"):
        afbouw_profiel = st.checkbox("Afbouw profiel o.b.v. horizon", value=False)
        
        if not afbouw_profiel:
            if "profiel_handmatig" in st.session_state and st.session_state["profiel_handmatig"] not in profiel_opties:
                del st.session_state["profiel_handmatig"]

            default_idx = min(2, len(profiel_opties) - 1)
            profiel = st.selectbox("Profiel", options=profiel_opties, index=default_idx, key="profiel_handmatig")
            profiel_input = profiel
        else:
            st.caption("Profiel wordt maandelijks defensiever.")
            if "profiel_start_afbouw" in st.session_state and st.session_state["profiel_start_afbouw"] not in profiel_opties:
                del st.session_state["profiel_start_afbouw"]

            default_start = default_profiel_for_horizon(int(horizon_jaren))
            default_index = profiel_opties.index(default_start) if default_start in profiel_opties else 0
            profiel = st.selectbox("Startprofiel", options=profiel_opties, index=default_index, key="profiel_start_afbouw")
            profiel_input = profiel
            
        n_scenarios = st.number_input("Aantal marktsimulaties", min_value=100, value=5000, step=100)

    with st.expander("Periodieke cashflows"):
        periodieke_storting = currency_text_input("Maandelijkse Storting (€)", key="periodieke_storting", default=0.0)
        storting_beperken = st.checkbox("Beperk storting tot periode")
        if storting_beperken and periodieke_storting > 0:
            periodieke_storting_startdatum = st.date_input("Start storting", value=startdatum, min_value=startdatum, max_value=end_date)
            periodieke_storting_einddatum = st.date_input("Eind storting", value=end_date, min_value=startdatum, max_value=end_date)
        else:
            periodieke_storting_startdatum = None
            periodieke_storting_einddatum = None
        
        st.divider()
        
        periodieke_onttrekking = currency_text_input("Maandelijkse Onttrekking (€)", key="periodieke_onttrekking", default=0.0)
        onttrekking_beperken = st.checkbox("Beperk onttrekking tot periode")
        if onttrekking_beperken and periodieke_onttrekking > 0:
            periodieke_onttrekking_startdatum = st.date_input("Start onttrekking", value=startdatum, min_value=startdatum, max_value=end_date)
            periodieke_onttrekking_einddatum = st.date_input("Eind onttrekking", value=end_date, min_value=startdatum, max_value=end_date)
        else:
            periodieke_onttrekking_startdatum = None
            periodieke_onttrekking_einddatum = None

    with st.expander("Eenmalige cashflows"):
        if st.session_state.eenmalige_cashflows:
            for idx, cf in enumerate(st.session_state.eenmalige_cashflows):
                cf_col1, cf_col2 = st.columns([3, 1])
                with cf_col1:
                    st.write(f"{'+' if cf.type == 'storting' else '-'} €{cf.bedrag:,.0f} op {cf.datum.strftime('%d-%m-%y')}")
                with cf_col2:
                    if st.button("Verwijder", key=f"delete_{idx}"):
                        st.session_state.eenmalige_cashflows.pop(idx)
                        st.rerun()
            st.divider()
            
        with st.form("add_cashflow_form", clear_on_submit=True):
            new_bedrag_raw = st.text_input("Bedrag (€)", value="0,00")
            new_type = st.selectbox("Type", options=["storting", "onttrekking"])
            new_datum = st.date_input("Datum", min_value=startdatum, max_value=end_date, value=startdatum)
            
            if st.form_submit_button("Toevoegen", use_container_width=True):
                try:
                    new_bedrag = parse_nl_number(new_bedrag_raw)
                    if new_bedrag > 0:
                        st.session_state.eenmalige_cashflows.append(EenmaligeCashflow(bedrag=new_bedrag, datum=new_datum, type=new_type))
                        st.rerun()
                except Exception:
                    st.error("Ongeldig bedrag.")

    eenmalige_cashflows_list = st.session_state.eenmalige_cashflows.copy()

    st.divider()
    calculate_clicked = st.button("Bereken", type="primary", use_container_width=True)

# ==========================================
# MAIN CONTENT: Calculations & Results
# ==========================================
st.title("Bloei rekenmodule")

inp = RekenInput(
    startvermogen=startvermogen,
    profiel=profiel_input,
    startdatum=startdatum,
    horizon_jaren=horizon_jaren,
    n_scenarios=n_scenarios,
    periodieke_storting_maandelijks=periodieke_storting,
    periodieke_onttrekking_maandelijks=periodieke_onttrekking,
    periodieke_storting_startdatum=periodieke_storting_startdatum,
    periodieke_storting_einddatum=periodieke_storting_einddatum,
    periodieke_onttrekking_startdatum=periodieke_onttrekking_startdatum,
    periodieke_onttrekking_einddatum=periodieke_onttrekking_einddatum,
    eenmalige_cashflows=eenmalige_cashflows_list,
    afbouw_profiel=afbouw_profiel,
)

if "reken_input" not in st.session_state:
    st.session_state.reken_input = inp

if calculate_clicked:
    st.session_state.reken_input = inp

current_inp = st.session_state.reken_input

@st.cache_data(show_spinner=False)
def get_berekening(reken_inp: RekenInput):
    return bereken_kosten(reken_inp)

result = get_berekening(current_inp)
active_startvermogen = current_inp.startvermogen

def fmt_eur(amount: float, decimals: int = 2) -> str:
    return fmt_eur_nl(amount, decimals)

def signed_class(amount: float) -> str:
    return "bloei-positive" if amount >= 0 else "bloei-negative"

# Top Metrics
st.subheader("Verwacht resultaat")
return_col1, return_col2, return_col3 = st.columns(3)

with return_col1:
    st.metric("Eindvermogen netto", fmt_eur(result.verwacht_eindvermogen_netto, 0))
    st.markdown(f"<p class='bloei-note'>Na aftrek van alle kosten</p>", unsafe_allow_html=True)

with return_col2:
    winst_netto = result.verwacht_eindvermogen_netto - (active_startvermogen + sum(result.tijdlijn_cashflow_netto))
    st.metric("Verwachte winst netto", fmt_eur(winst_netto, 0))
    st.markdown(f"<p class='bloei-note'>Puur rendement na kosten</p>", unsafe_allow_html=True)
    
with return_col3:
    st.metric("Totale impact kosten", fmt_eur(-float(result.totale_kosten_impact), 0))
    st.markdown(f"<p class='bloei-note'>Betaalde kosten + misgelopen rendement</p>", unsafe_allow_html=True)

st.divider()

# Dataframe Prep
df = pd.DataFrame({
    "datum": [datetime.combine(d, datetime.min.time()) for d in result.tijdlijn_datums],
    "vermogen_bruto": [float(v) for v in result.tijdlijn_vermogen_bruto],
    "vermogen_netto": [float(v) for v in result.tijdlijn_vermogen_netto],
    "vermogen_p10_netto": [float(v) for v in result.tijdlijn_vermogen_p10_netto],
    "vermogen_p90_netto": [float(v) for v in result.tijdlijn_vermogen_p90_netto],
    "cashflow_netto": [float(cf) for cf in result.tijdlijn_cashflow_netto],
    "kosten_cumulatief_betaald": [float(v) for v in result.tijdlijn_kosten_cumulatief],
    "profiel": list(result.tijdlijn_profiel),
})
df["kosten_impact_cumulatief"] = df["vermogen_bruto"] - df["vermogen_netto"]
df["kosten_misgelopen_rendement_cumulatief"] = (df["kosten_impact_cumulatief"] - df["kosten_cumulatief_betaald"]).clip(lower=0.0)
df["tooltip_datum"] = df["datum"].dt.strftime("%b %Y")
df["tooltip_bruto"] = df["vermogen_bruto"].map(lambda x: fmt_eur(x, 0))
df["tooltip_netto"] = df["vermogen_netto"].map(lambda x: fmt_eur(x, 0))
df["tooltip_p10"] = df["vermogen_p10_netto"].map(lambda x: fmt_eur(x, 0))
df["tooltip_p90"] = df["vermogen_p90_netto"].map(lambda x: fmt_eur(x, 0))


tab_vermogensopbouw, tab_waterfall, tab_cashflow, tab_kosten = st.tabs([
    "Vermogensopbouw", "Componenten", "Cashflow overzicht", "Kosten overzicht"
])

with tab_vermogensopbouw:
    if len(df) <= 1:
        st.info("Geen tijdlijn, horizon is 0 jaar.")
    else:
        # Toggles for chart
        toggles_col1, toggles_col2 = st.columns(2)
        with toggles_col1:
            show_p10_p90 = st.toggle("Toon onzekerheidsmarges (P10/P90)", value=True)
        with toggles_col2:
            show_bruto = st.toggle("Toon bruto ontwikkeling (zonder kosten)", value=True)
            
        base = alt.Chart(df)
        y_min = float(df[["vermogen_p10_netto", "vermogen_netto", "vermogen_bruto"]].min().min())
        y_max = float(df[["vermogen_p90_netto", "vermogen_netto", "vermogen_bruto"]].max().max())
        y_padding = max((y_max - y_min) * 0.12, max(1.0, y_max * 0.03))
        y_domain_min = max(0.0, y_min - y_padding)
        y_domain_max = y_max + y_padding
        first_tick = df["datum"].iloc[0].to_pydatetime()
        year_ticks = [datetime(year, 1, 1) for year in range(first_tick.year, df["datum"].iloc[-1].year + 1)]
        if first_tick not in year_ticks:
            year_ticks.insert(0, first_tick)
            
        x_year_axis = alt.X(
            "datum:T",
            title=None,
            axis=alt.Axis(format="%Y", values=year_ticks, labelAngle=0, grid=False),
        )
        
        y_axis = alt.Y(
            "vermogen_netto:Q", 
            title="Vermogen (€)",
            scale=alt.Scale(domain=[y_domain_min, y_domain_max], nice=False, zero=False),
            axis=alt.Axis(format=",.0f", labelExpr="replace(datum.label, regexp(',', 'g'), '.')", gridColor="#f0f0f0")
        )

        layers = []
        
        if show_p10_p90:
            band_netto = base.mark_area(opacity=0.15, color=BLOEI_PETROL).encode(
                x=x_year_axis,
                y=alt.Y("vermogen_p10_netto:Q"),
                y2=alt.Y2("vermogen_p90_netto:Q"),
                tooltip=[
                    alt.Tooltip("tooltip_datum:N", title="Maand"),
                    alt.Tooltip("tooltip_p90:N", title="Optimistisch (P90)"),
                    alt.Tooltip("tooltip_netto:N", title="Verwacht (Netto)"),
                    alt.Tooltip("tooltip_p10:N", title="Pessimistisch (P10)"),
                ]
            )
            layers.append(band_netto)

        if show_bruto:
            line_bruto = base.mark_line(color=BLOEI_PINK, strokeDash=[6, 4], strokeWidth=2).encode(
                x=x_year_axis,
                y=alt.Y("vermogen_bruto:Q"),
                tooltip=[
                    alt.Tooltip("tooltip_datum:N", title="Maand"),
                    alt.Tooltip("tooltip_bruto:N", title="Bruto Vermogen"),
                ]
            )
            layers.append(line_bruto)

        line_netto = base.mark_line(
            color=BLOEI_PETROL, 
            strokeWidth=4
        ).encode(
            x=x_year_axis,
            y=y_axis,
            tooltip=[
                alt.Tooltip("tooltip_datum:N", title="Maand"),
                alt.Tooltip("tooltip_netto:N", title="Netto Vermogen"),
                alt.Tooltip("profiel:N", title="Risicoprofiel"),
            ]
        )
        layers.append(line_netto)
        
        vermogen_chart = alt.layer(*layers).properties(height=500).configure_view(strokeWidth=0)
        st.altair_chart(vermogen_chart, use_container_width=True)

with tab_waterfall:
    st.subheader("Van start tot eindvermogen")
    st.write("Hoe uw vermogen is opgebouwd over de gehele periode.")
    
    totale_stortingen = sum(cf for cf in result.tijdlijn_cashflow_netto if cf > 0)
    totale_onttrekkingen = sum(cf for cf in result.tijdlijn_cashflow_netto if cf < 0)
    
    bruto_rendement = result.verwacht_eindvermogen_bruto - (active_startvermogen + totale_stortingen + totale_onttrekkingen)
    kosten_impact = -(result.verwacht_eindvermogen_bruto - result.verwacht_eindvermogen_netto)
    
    rows = []
    current_idx = 1
    
    # 1. Start
    components_so_far = [("Startvermogen", active_startvermogen, 1)]
    for comp, amt, ord_idx in components_so_far:
        rows.append({"category": f"{current_idx}. Start", "component": comp, "amount": amt, "order_idx": ord_idx})
    current_idx += 1
    
    # 2. + Inleg
    if totale_stortingen > 0:
        components_so_far.append(("Stortingen", totale_stortingen, 2))
        for comp, amt, ord_idx in components_so_far:
            rows.append({"category": f"{current_idx}. + Inleg", "component": comp, "amount": amt, "order_idx": ord_idx})
        current_idx += 1
        
    # 3. + Rendement (Bruto)
    components_so_far.append(("Rendement (Bruto)", bruto_rendement, 3))
    for comp, amt, ord_idx in components_so_far:
        rows.append({"category": f"{current_idx}. + Rendement (Bruto)", "component": comp, "amount": amt, "order_idx": ord_idx})
    current_idx += 1
        
    # 4. - Onttrekkingen
    if totale_onttrekkingen < 0:
        components_so_far.append(("Onttrekkingen", totale_onttrekkingen, 4))
        for comp, amt, ord_idx in components_so_far:
            rows.append({"category": f"{current_idx}. - Onttrekkingen", "component": comp, "amount": amt, "order_idx": ord_idx})
        current_idx += 1
        
    # 5. - Kosten (Eindvermogen)
    components_so_far.append(("Kosten Impact", kosten_impact, 5))
    for comp, amt, ord_idx in components_so_far:
        rows.append({"category": f"{current_idx}. Eindvermogen (Netto)", "component": comp, "amount": amt, "order_idx": ord_idx})

    df_rows = []
    categories = []
    for r in rows:
        if r["category"] not in categories:
            categories.append(r["category"])
            
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_rows = sorted(cat_rows, key=lambda x: x["order_idx"])
        current_y = 0.0
        for r in cat_rows:
            y1 = current_y
            y2 = current_y + r["amount"]
            r["y_start"] = y1
            r["y_end"] = y2
            current_y = y2
            df_rows.append(r)

    wf_data = pd.DataFrame(df_rows)
    wf_data['tooltip_amount'] = wf_data['amount'].map(lambda x: fmt_eur(x, 0))
    
    domain = ["Startvermogen"]
    range_ = [BLOEI_PETROL]
    
    if totale_stortingen > 0:
        domain.append("Stortingen")
        range_.append(POSITIVE_COLOR)
        
    if totale_onttrekkingen < 0:
        domain.append("Onttrekkingen")
        range_.append(NEGATIVE_COLOR)
        
    domain.append("Rendement (Bruto)")
    range_.append(BLOEI_PINK)
    
    domain.append("Kosten Impact")
    range_.append("#d65c60")
    
    color_scale = alt.Scale(
        domain=domain,
        range=range_
    )
    
    waterfall_chart = alt.Chart(wf_data).mark_bar(cornerRadius=4, size=50).encode(
        x=alt.X("category:N", title=None, axis=alt.Axis(labels=False, tickSize=0)),
        y=alt.Y("y_start:Q", title="Vermogen (€)", axis=alt.Axis(format=",.0f", labelExpr="replace(datum.label, regexp(',', 'g'), '.')")),
        y2=alt.Y2("y_end:Q"),
        color=alt.Color("component:N", scale=color_scale, legend=alt.Legend(title="Component", orient="bottom")),
        order=alt.Order("order_idx:Q", sort="ascending"),
        tooltip=[
            alt.Tooltip("category:N", title="Fase"),
            alt.Tooltip("component:N", title="Onderdeel"),
            alt.Tooltip("tooltip_amount:N", title="Bedrag"),
        ]
    ).properties(height=450).configure_view(strokeWidth=0)
    
    st.altair_chart(waterfall_chart, use_container_width=True)

with tab_cashflow:
    if len(df) > 1:
        df_work = df.copy()
        df_work["maand_index"] = range(len(df_work))
        jaar_nul = df_work.iloc[[0]].copy()
        jaar_nul["Jaar"] = 0
        jaar_nul["cashflow_netto"] = 0.0

        df_periodiek = df_work[df_work["maand_index"] > 0].copy()
        df_periodiek["Jaar"] = ((df_periodiek["maand_index"] - 1) // 12) + 1

        yearly_end = df_periodiek.groupby("Jaar", as_index=False).agg(
            vermogen_bruto=("vermogen_bruto", "last"),
            vermogen_netto=("vermogen_netto", "last"),
            kosten_cumulatief_betaald=("kosten_cumulatief_betaald", "last"),
            kosten_misgelopen_rendement_cumulatief=("kosten_misgelopen_rendement_cumulatief", "last"),
            profiel=("profiel", "last"),
        )
        yearly_cashflow = df_periodiek.groupby("Jaar", as_index=False).agg(cashflow_netto=("cashflow_netto", "sum"))
        df_year = yearly_end.merge(yearly_cashflow, on="Jaar", how="left")

        jaar_nul_row = pd.DataFrame({
            "Jaar": [0],
            "vermogen_bruto": [float(jaar_nul["vermogen_bruto"].iloc[0])],
            "vermogen_netto": [float(jaar_nul["vermogen_netto"].iloc[0])],
            "cashflow_netto": [0.0],
            "kosten_cumulatief_betaald": [0.0],
            "kosten_misgelopen_rendement_cumulatief": [0.0],
            "profiel": [jaar_nul["profiel"].iloc[0]],
        })

        df_year = pd.concat([jaar_nul_row, df_year], ignore_index=True)
        df_year["Cumulatieve kosten impact"] = df_year["vermogen_bruto"] - df_year["vermogen_netto"]

        for col in ["vermogen_bruto", "vermogen_netto", "cashflow_netto", "kosten_cumulatief_betaald", "kosten_misgelopen_rendement_cumulatief", "Cumulatieve kosten impact"]:
            df_year[col] = df_year[col].map(lambda x: fmt_eur(x, 0))

        st.dataframe(
            df_year[[
                "Jaar", "vermogen_bruto", "vermogen_netto", "cashflow_netto", 
                "kosten_cumulatief_betaald", "kosten_misgelopen_rendement_cumulatief", 
                "Cumulatieve kosten impact", "profiel"
            ]].rename(columns={
                "vermogen_bruto": "Vermogen bruto",
                "vermogen_netto": "Vermogen netto",
                "cashflow_netto": "Netto cashflow",
                "kosten_cumulatief_betaald": "Cumul. kosten betaald",
                "kosten_misgelopen_rendement_cumulatief": "Cumul. misgelopen rendement",
                "Cumulatieve kosten impact": "Cumulatieve kosten impact",
                "profiel": "Profiel (eind jaar)"
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Geen cashflowtabel, horizon is 0 jaar.")

with tab_kosten:
    kosten_left, kosten_center, kosten_right = st.columns([1, 4, 1])
    with kosten_center:
        st.markdown(
            f"""
<h3 class="bloei-title-petrol">Kosten overzicht</h3>
<table class="kosten-open-table">
  <thead>
    <tr>
      <th>Kostencomponent</th>
      <th>Gemiddeld per jaar (%)</th>
      <th>Totaal in Euro's</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Beheerfee</td>
      <td>{fmt_pct_nl(result.gemiddelde_beheerkosten_pct)}</td>
      <td>{fmt_eur(-float(result.totale_beheerkosten_betaald), 0)}</td>
    </tr>
    <tr>
      <td>Fondskosten</td>
      <td>{fmt_pct_nl(result.gemiddelde_fondskosten_pct)}</td>
      <td>{fmt_eur(-float(result.totale_fondskosten_betaald), 0)}</td>
    </tr>
    <tr>
      <td>Spread kosten</td>
      <td>{fmt_pct_nl(result.gemiddelde_spreadkosten_pct)}</td>
      <td>{fmt_eur(-float(result.totale_spreadkosten_betaald), 0)}</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Totaal betaald</strong></td>
      <td><strong>{fmt_pct_nl(result.gemiddelde_totale_kosten_pct)}</strong></td>
      <td><strong>{fmt_eur(-float(result.totale_kosten_betaald), 0)}</strong></td>
    </tr>
  </tbody>
</table>
<p class="bloei-note" style="text-align: right; margin-top: 1rem;">
  <em>Totale kosten impact inclusief misgelopen rendement: <strong>{fmt_eur(-float(result.totale_kosten_impact), 0)}</strong></em>
</p>
""",
            unsafe_allow_html=True,
        )
