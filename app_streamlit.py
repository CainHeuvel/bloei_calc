"""Streamlit UI for the Bloei Rekenmodule."""

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
    page_title="Bloei Rekenmodule",
    page_icon="📊",
    layout="wide",
)

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
div[data-testid="metric-container"] {{
    background-color: var(--bloei-warmgrey);
    border: 1px solid rgba(15, 73, 79, 0.2);
    border-radius: 8px;
    padding: 1rem;
}}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: var(--bloei-petrol);
}}
.bloei-note {{
    margin: 0.35rem 0 0;
    font-size: 0.95rem;
}}
.bloei-positive {{
    color: var(--bloei-positive);
    font-weight: 600;
}}
.bloei-negative {{
    color: var(--bloei-negative);
    font-weight: 600;
}}
.kosten-open-table {{
    width: min(100%, 760px);
    margin: 0.75rem auto 0;
    border-collapse: collapse;
    color: var(--bloei-petrol);
}}
.kosten-open-table th {{
    text-align: left;
    font-size: 0.92rem;
    font-weight: 600;
    padding: 0.5rem 0.35rem;
    border-bottom: 1px solid rgba(15, 73, 79, 0.25);
}}
.kosten-open-table td {{
    padding: 0.55rem 0.35rem;
    border-bottom: 1px solid rgba(15, 73, 79, 0.15);
}}
.kosten-open-table th:last-child,
.kosten-open-table td:last-child {{
    text-align: right;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Bloei Rekenmodule")

st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    startvermogen = currency_text_input(
        "Startvermogen (€)",
        key="startvermogen",
        default=100000.0,
        min_value=0.01,
    )

with col2:
    startdatum = st.date_input(
        "Startdatum",
        value=date.today(),
    )
    
    horizon_jaren = st.slider(
        "Horizon (jaren)",
        min_value=0,
        max_value=40,
        value=10,
    )
    

with st.expander("Geavanceerde Instellingen"):
    afbouw_profiel = st.checkbox(
        "Afbouw profiel op basis van horizon",
        value=False,
    )
    n_scenarios = st.number_input(
        "Aantal marktsimulaties",
        min_value=1,
        value=5000,
        step=100,
    )

end_date = _add_years(startdatum, int(horizon_jaren))

st.divider()
st.subheader("Profiel")

profiel_opties = allowed_profielen_for_horizon(int(horizon_jaren))

if not afbouw_profiel:
    if "profiel_handmatig" in st.session_state and st.session_state["profiel_handmatig"] not in profiel_opties:
        del st.session_state["profiel_handmatig"]

    default_idx = min(2, len(profiel_opties) - 1)
    profiel = st.selectbox(
        "Profiel",
        options=profiel_opties,
        index=default_idx,
        key="profiel_handmatig",
    )
    profiel_input = profiel
    profiel_label = "Vast profiel"
else:
    st.caption("Afbouw: profiel wordt per maand defensiever op basis van resterende horizon.")
    
    if "profiel_start_afbouw" in st.session_state and st.session_state["profiel_start_afbouw"] not in profiel_opties:
        del st.session_state["profiel_start_afbouw"]

    default_start = default_profiel_for_horizon(int(horizon_jaren))
    default_index = profiel_opties.index(default_start) if default_start in profiel_opties else 0

    profiel = st.selectbox(
        "Startprofiel",
        options=profiel_opties,
        index=default_index,
        key="profiel_start_afbouw",
    )
    profiel_input = profiel
    profiel_label = "Startprofiel (afbouw-cap)"

st.divider()
st.header("Cashflows")
cashflow_col1, cashflow_col2 = st.columns(2)

with cashflow_col1:
    st.subheader("Periodieke Cashflows")
    periodieke_storting = currency_text_input(
        "Periodieke Storting (EUR/maand)",
        key="periodieke_storting",
        default=0.0,
        min_value=0.0,
    )

    storting_beperken = st.checkbox("Beperk periodieke storting tot periode")
    if storting_beperken and periodieke_storting > 0:
        st_col_s, st_col_e = st.columns(2)
        with st_col_s:
            periodieke_storting_startdatum = st.date_input("Startdatum storting", value=startdatum, min_value=startdatum, max_value=end_date)
        with st_col_e:
            periodieke_storting_einddatum = st.date_input("Einddatum storting", value=end_date, min_value=startdatum, max_value=end_date)
    else:
        periodieke_storting_startdatum = None
        periodieke_storting_einddatum = None
    
    periodieke_onttrekking = currency_text_input(
        "Periodieke Onttrekking (EUR/maand)",
        key="periodieke_onttrekking",
        default=0.0,
        min_value=0.0,
    )

    onttrekking_beperken = st.checkbox("Beperk periodieke onttrekking tot periode")
    if onttrekking_beperken and periodieke_onttrekking > 0:
        ot_col_s, ot_col_e = st.columns(2)
        with ot_col_s:
            periodieke_onttrekking_startdatum = st.date_input("Startdatum onttrekking", value=startdatum, min_value=startdatum, max_value=end_date)
        with ot_col_e:
            periodieke_onttrekking_einddatum = st.date_input("Einddatum onttrekking", value=end_date, min_value=startdatum, max_value=end_date)
    else:
        periodieke_onttrekking_startdatum = None
        periodieke_onttrekking_einddatum = None

with cashflow_col2:
    st.subheader("Eenmalige Cashflows")
    
    if st.session_state.eenmalige_cashflows:
        st.write("**Huidige eenmalige cashflows:**")
        for idx, cf in enumerate(st.session_state.eenmalige_cashflows):
            row_col1, row_col2, row_col3, row_col4 = st.columns([2, 2, 2, 1])
            with row_col1:
                st.write(f"€{cf.bedrag:,.2f}")
            with row_col2:
                st.write("Storting" if cf.type == 'storting' else "Onttrekking")
            with row_col3:
                st.write(cf.datum.strftime("%d-%m-%Y"))
            with row_col4:
                if st.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.eenmalige_cashflows.pop(idx)
                    st.rerun()
        st.divider()
    
    st.write("**Nieuwe cashflow toevoegen:**")
    with st.form("add_cashflow_form", clear_on_submit=True):
        new_col1, new_col2, new_col3, new_col4 = st.columns([2.5, 2, 2.5, 1])
        new_bedrag = 0.0
        with new_col1:
            new_bedrag_raw = st.text_input("1. Bedrag (€)", value="0,00")
        with new_col2:
            new_type = st.selectbox("2. Type", options=["storting", "onttrekking"])
        with new_col3:
            new_datum = st.date_input("3. Wanneer", min_value=startdatum, max_value=end_date, value=startdatum)
        with new_col4:
            st.write("")
            st.write("")
            add_button = st.form_submit_button("➕", use_container_width=True)
        
        if add_button:
            try:
                new_bedrag = parse_nl_number(new_bedrag_raw)
            except Exception:
                st.error("Voer een geldig bedrag in, bijvoorbeeld 100.000,00.")
                new_bedrag = 0.0

            if new_bedrag > 0:
                st.session_state.eenmalige_cashflows.append(EenmaligeCashflow(bedrag=new_bedrag, datum=new_datum, type=new_type))
                st.rerun()
    
    eenmalige_cashflows_list = st.session_state.eenmalige_cashflows.copy()

st.divider()
if st.button("Bereken Prognose", type="primary", use_container_width=True):
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
    
    result = bereken_kosten(inp)
    
    st.header("Verwachte Vermogensontwikkeling")
    
    st.subheader("Resultaten op basis van historische marktsimulaties")
    return_col1, return_col2, return_col3, return_col4 = st.columns(4)

    def fmt_eur(amount: float, decimals: int = 2) -> str:
        return fmt_eur_nl(amount, decimals)
    
    def signed_class(amount: float) -> str:
        return "bloei-positive" if amount >= 0 else "bloei-negative"

    with return_col1:
        st.metric("Eindvermogen Bruto (zonder kosten)", fmt_eur(result.verwacht_eindvermogen_bruto))
        st.caption("Fictieve ontwikkeling puur op basis van marktrendement.")
        st.markdown(
            f"<p class='bloei-note'>Bruto eindwaarde: "
            f"<span class='{signed_class(result.verwacht_eindvermogen_bruto)}'>"
            f"{fmt_eur(result.verwacht_eindvermogen_bruto)}</span></p>",
            unsafe_allow_html=True,
        )
        
    with return_col2:
        st.metric("Eindvermogen Netto (na kosten)", fmt_eur(result.verwacht_eindvermogen_netto))
        st.caption(f"P10: {fmt_eur(result.verwacht_eindvermogen_p10_netto)} | P50: {fmt_eur(result.verwacht_eindvermogen_p50_netto)} | P90: {fmt_eur(result.verwacht_eindvermogen_p90_netto)}")
        st.markdown(
            f"<p class='bloei-note'>Verwachte netto eindwaarde: "
            f"<span class='{signed_class(result.verwacht_eindvermogen_netto)}'>"
            f"{fmt_eur(result.verwacht_eindvermogen_netto)}</span></p>",
            unsafe_allow_html=True,
        )
        
    with return_col3:
        st.metric("Totale Cumulatieve Kosten", fmt_eur(-float(result.totale_kosten_betaald)))
        st.caption("Exclusief misgelopen rendement op rendement.")
        st.markdown(
            f"<p class='bloei-note'>Betaalde kosten over de periode: "
            f"<span class='{signed_class(-float(result.totale_kosten_betaald))}'>"
            f"{fmt_eur(-float(result.totale_kosten_betaald))}</span></p>",
            unsafe_allow_html=True,
        )

    with return_col4:
        st.metric("Totale Impact Kosten", fmt_eur(-float(result.totale_kosten_impact)))
        st.caption("Cumulatieve kosten + misgelopen rendement op rendement.")
        st.markdown(
            f"<p class='bloei-note'>Totale kostenimpact over de periode: "
            f"<span class='{signed_class(-float(result.totale_kosten_impact))}'>"
            f"{fmt_eur(-float(result.totale_kosten_impact))}</span></p>",
            unsafe_allow_html=True,
        )

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
    df["kosten_misgelopen_rendement_cumulatief"] = (
        df["kosten_impact_cumulatief"] - df["kosten_cumulatief_betaald"]
    ).clip(lower=0.0)
    df["tooltip_datum"] = df["datum"].dt.strftime("%b %Y")
    df["tooltip_bruto"] = df["vermogen_bruto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_netto"] = df["vermogen_netto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_p10"] = df["vermogen_p10_netto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_p90"] = df["vermogen_p90_netto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_kosten_betaald"] = df["kosten_cumulatief_betaald"].map(lambda x: fmt_eur(-x, 2))
    df["tooltip_kosten_impact"] = df["kosten_impact_cumulatief"].map(lambda x: fmt_eur(-x, 2))
    df["tooltip_kosten_misgelopen"] = df["kosten_misgelopen_rendement_cumulatief"].map(lambda x: fmt_eur(-x, 2))

    st.divider()
    st.subheader("Tijdlijn")
    tab_vermogensopbouw, tab_cashflow, tab_kosten = st.tabs(["Vermogensopbouw", "Cashflow", "Kosten"])

    with tab_vermogensopbouw:
        if len(df) <= 1:
            st.info("Geen tijdlijn, horizon is 0 jaar.")
        else:
            chart_left, chart_center, chart_right = st.columns([1.2, 4.6, 1.2])
            with chart_center:
                base = alt.Chart(df)
                y_min = float(df[["vermogen_p10_netto", "vermogen_netto", "vermogen_bruto"]].min().min())
                y_max = float(df[["vermogen_p90_netto", "vermogen_netto", "vermogen_bruto"]].max().max())
                y_padding = max((y_max - y_min) * 0.12, max(1.0, y_max * 0.03))
                y_domain_min = max(0.0, y_min - y_padding)
                y_domain_max = y_max + y_padding
                chart_height = min(560, max(430, 300 + int(len(df) * 2.2)))
                first_tick = df["datum"].iloc[0].to_pydatetime()
                year_ticks = [datetime(year, 1, 1) for year in range(first_tick.year, df["datum"].iloc[-1].year + 1)]
                if first_tick not in year_ticks:
                    year_ticks.insert(0, first_tick)
                x_year_axis = alt.X(
                    "datum:T",
                    title="Jaar",
                    axis=alt.Axis(format="%Y", values=year_ticks, labelAngle=0),
                )
                
                # Netto onzekerheidsband
                band_netto = base.mark_area(opacity=0.15, color=BLOEI_PETROL).encode(
                    x=x_year_axis,
                    y=alt.Y(
                        "vermogen_p10_netto:Q",
                        title="Vermogen",
                        scale=alt.Scale(domain=[y_domain_min, y_domain_max], nice=False, zero=False),
                        axis=alt.Axis(format=",.0f", labelExpr="'€ ' + replace(datum.label, regexp(',', 'g'), '.')"),
                    ),
                    y2=alt.Y2("vermogen_p90_netto:Q"),
                    tooltip=[
                        alt.Tooltip("tooltip_datum:N", title="Periode"),
                        alt.Tooltip("tooltip_p10:N", title="P10"),
                        alt.Tooltip("tooltip_p90:N", title="P90"),
                    ],
                )
                
                # Lijnen: Bruto en Netto
                line_bruto = base.mark_line(color=BLOEI_PINK, strokeDash=[6, 4], strokeWidth=2).encode(
                    x=x_year_axis,
                    y=alt.Y("vermogen_bruto:Q", scale=alt.Scale(domain=[y_domain_min, y_domain_max], nice=False, zero=False)),
                    tooltip=[
                        alt.Tooltip("tooltip_datum:N", title="Periode"),
                        alt.Tooltip("tooltip_bruto:N", title="Bruto"),
                    ],
                )
                line_netto = base.mark_line(color=BLOEI_PETROL, strokeWidth=4).encode(
                    x=x_year_axis,
                    y=alt.Y("vermogen_netto:Q", scale=alt.Scale(domain=[y_domain_min, y_domain_max], nice=False, zero=False)),
                    tooltip=[
                        alt.Tooltip("tooltip_datum:N", title="Periode"),
                        alt.Tooltip("tooltip_netto:N", title="Netto"),
                        alt.Tooltip("profiel:N", title="Profiel"),
                    ],
                )
                
                vermogen_chart = (band_netto + line_bruto + line_netto).properties(height=chart_height)
                
                st.altair_chart(vermogen_chart, use_container_width=True)
                st.caption("Roze gestippeld: Bruto ontwikkeling | Petrol lijn + waaier: Netto ontwikkeling (P10/P90)")

    with tab_cashflow:
        if len(df) > 1:
            df_work = df.copy()
            df_work["maand_index"] = range(len(df_work))

            jaar_nul = df_work.iloc[[0]].copy()
            jaar_nul["Jaar"] = 0
            jaar_nul["cashflow_netto"] = 0.0

            df_periodiek = df_work[df_work["maand_index"] > 0].copy()
            df_periodiek["Jaar"] = ((df_periodiek["maand_index"] - 1) // 12) + 1

            yearly_end = (
                df_periodiek.groupby("Jaar", as_index=False)
                .agg(
                    vermogen_bruto=("vermogen_bruto", "last"),
                    vermogen_netto=("vermogen_netto", "last"),
                    kosten_cumulatief_betaald=("kosten_cumulatief_betaald", "last"),
                    kosten_misgelopen_rendement_cumulatief=("kosten_misgelopen_rendement_cumulatief", "last"),
                    profiel=("profiel", "last"),
                )
            )
            yearly_cashflow = (
                df_periodiek.groupby("Jaar", as_index=False)
                .agg(cashflow_netto=("cashflow_netto", "sum"))
            )

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
            df_year["Cumulatieve Kosten Impact"] = df_year["vermogen_bruto"] - df_year["vermogen_netto"]

            for col in [
                "vermogen_bruto",
                "vermogen_netto",
                "cashflow_netto",
                "kosten_cumulatief_betaald",
                "kosten_misgelopen_rendement_cumulatief",
                "Cumulatieve Kosten Impact",
            ]:
                df_year[col] = df_year[col].map(lambda x: fmt_eur(x, 0))

            st.dataframe(
                df_year[
                    [
                        "Jaar",
                        "vermogen_bruto",
                        "vermogen_netto",
                        "cashflow_netto",
                        "kosten_cumulatief_betaald",
                        "kosten_misgelopen_rendement_cumulatief",
                        "Cumulatieve Kosten Impact",
                        "profiel",
                    ]
                ].rename(
                    columns={
                        "vermogen_bruto": "Vermogen Bruto",
                        "vermogen_netto": "Vermogen Netto",
                        "cashflow_netto": "Netto Cashflow",
                        "kosten_cumulatief_betaald": "Cumulatieve Kosten (excl. misgelopen rendement)",
                        "kosten_misgelopen_rendement_cumulatief": "Cumulatief Misgelopen Rendement op Kosten",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Geen cashflowtabel, horizon is 0 jaar.")

    with tab_kosten:
        kosten_left, kosten_center, kosten_right = st.columns([1.2, 4.6, 1.2])
        with kosten_center:
            st.markdown(
                f"""
<h3 style="color:{BLOEI_PETROL}; margin:0 0 0.2rem;">Gemiddelde jaarlijkse kosten</h3>
<table class="kosten-open-table">
  <thead>
    <tr>
      <th>Kostencomponent</th>
      <th>Gemiddeld jaarlijks</th>
      <th>Cumulatief gemiddeld</th>
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
    <tr>
      <td><strong>Totaal</strong></td>
      <td><strong>{fmt_pct_nl(result.gemiddelde_totale_kosten_pct)}</strong></td>
      <td><strong>{fmt_eur(-float(result.totale_kosten_betaald), 0)}</strong></td>
    </tr>
  </tbody>
</table>
""",
                unsafe_allow_html=True,
            )
