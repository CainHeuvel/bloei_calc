"""Streamlit UI for the Bloei Rekenmodule."""

import streamlit as st
from datetime import date, datetime
from bloei_rekenmodel import RekenInput, bereken_kosten, EenmaligeCashflow
import altair as alt
import pandas as pd

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

st.title("Bloei Rekenmodule")

st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    startvermogen = st.number_input(
        "Startvermogen (€)",
        min_value=0.01,
        value=100000.0,
        step=1000.0,
        format="%.2f",
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
    periodieke_storting = st.number_input(
        "Periodieke Storting (EUR/maand)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.2f",
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
    
    periodieke_onttrekking = st.number_input(
        "Periodieke Onttrekking (EUR/maand)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.2f",
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
        with new_col1:
            new_bedrag = st.number_input("1. Bedrag (€)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        with new_col2:
            new_type = st.selectbox("2. Type", options=["storting", "onttrekking"])
        with new_col3:
            new_datum = st.date_input("3. Wanneer", min_value=startdatum, max_value=end_date, value=startdatum)
        with new_col4:
            st.write("")
            st.write("")
            add_button = st.form_submit_button("➕", use_container_width=True)
        
        if add_button and new_bedrag > 0:
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
    return_col1, return_col2, return_col3 = st.columns(3)

    def fmt_eur(amount: float, decimals: int = 2) -> str:
        us = f"{amount:,.{decimals}f}"
        return "€ " + us.replace(",", "_").replace(".", ",").replace("_", ".")
    
    with return_col1:
        st.metric("Eindvermogen Bruto (zonder kosten)", fmt_eur(result.verwacht_eindvermogen_bruto))
        st.caption("Fictieve ontwikkeling puur op basis van marktrendement.")
        st.info(f"Bruto eindwaarde: {fmt_eur(result.verwacht_eindvermogen_bruto)}")
        
    with return_col2:
        st.metric("Eindvermogen Netto (na kosten)", fmt_eur(result.verwacht_eindvermogen_netto))
        st.caption(f"P10: {fmt_eur(result.verwacht_eindvermogen_p10_netto)} | P50: {fmt_eur(result.verwacht_eindvermogen_p50_netto)} | P90: {fmt_eur(result.verwacht_eindvermogen_p90_netto)}")
        st.success(f"Verwachte netto eindwaarde: {fmt_eur(result.verwacht_eindvermogen_netto)}")
        
    with return_col3:
        st.metric("Totale Impact Kosten", f"- {fmt_eur(result.totale_kosten_impact)}")
        st.caption("Cumulatieve kosten + misgelopen rendement op rendement.")
        st.info(f"Totale kostenimpact over de periode: - {fmt_eur(result.totale_kosten_impact)}")

    df = pd.DataFrame({
        "datum": [datetime.combine(d, datetime.min.time()) for d in result.tijdlijn_datums],
        "vermogen_bruto": [float(v) for v in result.tijdlijn_vermogen_bruto],
        "vermogen_netto": [float(v) for v in result.tijdlijn_vermogen_netto],
        "vermogen_p10_netto": [float(v) for v in result.tijdlijn_vermogen_p10_netto],
        "vermogen_p90_netto": [float(v) for v in result.tijdlijn_vermogen_p90_netto],
        "cashflow_netto": [float(cf) for cf in result.tijdlijn_cashflow_netto],
        "profiel": list(result.tijdlijn_profiel),
    })
    df["tooltip_datum"] = df["datum"].dt.strftime("%b %Y")
    df["tooltip_bruto"] = df["vermogen_bruto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_netto"] = df["vermogen_netto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_p10"] = df["vermogen_p10_netto"].map(lambda x: fmt_eur(x, 2))
    df["tooltip_p90"] = df["vermogen_p90_netto"].map(lambda x: fmt_eur(x, 2))

    st.divider()
    st.subheader("Tijdlijn")
    tab_grafieken, tab_tabel = st.tabs(["Grafieken", "Tabellen"])

    with tab_grafieken:
        if len(df) <= 1:
            st.info("Geen tijdlijn, horizon is 0 jaar.")
        else:
            base = alt.Chart(df)
            
            # Netto onzekerheidsband
            band_netto = base.mark_area(opacity=0.12, color="#6EA8FE").encode(
                x=alt.X("datum:T", title="Datum"),
                y=alt.Y(
                    "vermogen_p10_netto:Q",
                    title="Vermogen",
                    axis=alt.Axis(format=",.0f", labelExpr="'€ ' + replace(datum.label, ',' , '.')"),
                ),
                y2=alt.Y2("vermogen_p90_netto:Q"),
                tooltip=[
                    alt.Tooltip("tooltip_datum:N", title="Periode"),
                    alt.Tooltip("tooltip_p10:N", title="P10"),
                    alt.Tooltip("tooltip_p90:N", title="P90"),
                ],
            )
            
            # Lijnen: Bruto en Netto
            line_bruto = base.mark_line(color="#F59E0B", strokeDash=[4, 4], strokeWidth=2).encode(
                x=alt.X("datum:T"),
                y=alt.Y("vermogen_bruto:Q"),
                tooltip=[
                    alt.Tooltip("tooltip_datum:N", title="Periode"),
                    alt.Tooltip("tooltip_bruto:N", title="Bruto"),
                ],
            )
            line_netto = base.mark_line(color="#2563EB", strokeWidth=4).encode(
                x=alt.X("datum:T"),
                y=alt.Y("vermogen_netto:Q"),
                tooltip=[
                    alt.Tooltip("tooltip_datum:N", title="Periode"),
                    alt.Tooltip("tooltip_netto:N", title="Netto"),
                    alt.Tooltip("profiel:N", title="Profiel"),
                ],
            )
            
            vermogen_chart = (band_netto + line_bruto + line_netto).properties(height=350)
            
            st.altair_chart(vermogen_chart, use_container_width=True)
            st.caption("🟠 Oranje gestippeld: Bruto ontwikkeling | 🔵 Blauwe lijn: Netto ontwikkeling (inclusief P10/P90 onzekerheid)")

    with tab_tabel:
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
                "profiel": [jaar_nul["profiel"].iloc[0]],
            })

            df_year = pd.concat([jaar_nul_row, df_year], ignore_index=True)
            df_year["Cumulatieve Kosten Impact"] = df_year["vermogen_bruto"] - df_year["vermogen_netto"]

            for col in ["vermogen_bruto", "vermogen_netto", "cashflow_netto", "Cumulatieve Kosten Impact"]:
                df_year[col] = df_year[col].map(lambda x: fmt_eur(x, 0))

            st.table(df_year[["Jaar", "vermogen_bruto", "vermogen_netto", "cashflow_netto", "Cumulatieve Kosten Impact", "profiel"]])
