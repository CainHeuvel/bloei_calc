"""Pure calculation logic for the Bloei Rekenmodule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt

import numpy as np

from bloei_rekenmodel.domain import EenmaligeCashflow, RekenInput, RekenOutput


def _add_years(d: date, years: int) -> date:
    """Add years to a date, clamping Feb 29 -> Feb 28 when needed."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)


def _add_months(d: date, months: int) -> date:
    """Add months to a date while preserving day as much as possible."""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = d.day
    for candidate_day in (28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        if candidate_day > day:
            continue
        try:
            return date(year, month, candidate_day)
        except ValueError:
            continue
    return date(year, month, 1)


def _month_index_bucket(start: date, event: date) -> int:
    """Bucket date to 1-based month index by calendar month (year + month)."""
    return (event.year - start.year) * 12 + (event.month - start.month) + 1


def _clamp_period_to_month_indices(
    *,
    startdatum: date,
    enddatum_horizon: date,
    total_months: int,
    periode_start: date | None,
    periode_eind: date | None,
) -> tuple[int, int]:
    """Convert optional date range to inclusive month indices in [1, total_months]."""
    if total_months <= 0:
        return (1, 0)

    s = periode_start or startdatum
    e = periode_eind or enddatum_horizon
    if s < startdatum:
        s = startdatum
    if e > enddatum_horizon:
        e = enddatum_horizon

    start_idx = max(1, min(total_months, _month_index_bucket(startdatum, s)))
    end_idx = max(1, min(total_months, _month_index_bucket(startdatum, e)))
    if end_idx < start_idx:
        end_idx = start_idx
    return (start_idx, end_idx)


def _profiel_for_remaining_years(remaining_years: float) -> str:
    if remaining_years > 14:
        return "Zeer offensief"
    if remaining_years >= 10:
        return "Offensief"
    if remaining_years >= 8:
        return "Neutraal"
    if remaining_years >= 6:
        return "Matig defensief"
    if remaining_years > 3:
        return "Defensief"
    return "Niet beleggen"


_PROFIEL_ORDER_MOST_OFFENSIVE_TO_DEFENSIVE = [
    "Zeer offensief",
    "Offensief",
    "Neutraal",
    "Matig defensief",
    "Defensief",
    "Niet beleggen",
]


def _more_defensive_profiel(a: str, b: str) -> str:
    rank = {p: i for i, p in enumerate(_PROFIEL_ORDER_MOST_OFFENSIVE_TO_DEFENSIVE)}
    ra = rank.get(a, rank["Niet beleggen"])
    rb = rank.get(b, rank["Niet beleggen"])
    return a if ra >= rb else b


def _profiel_with_afbouw(start_profiel: str, total_months: int, month_index: int) -> str:
    remaining_years = max(0.0, (total_months - (month_index - 1)) / 12.0)
    auto_profiel = _profiel_for_remaining_years(remaining_years)
    profiel_maand = _more_defensive_profiel(start_profiel, auto_profiel)
    if remaining_years > 3.0 and profiel_maand == "Niet beleggen":
        return "Defensief"
    return profiel_maand


def _safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _safe_stat_mean(values: np.ndarray) -> float:
    return _safe_float(float(np.mean(values)))


def _safe_stat_percentile(values: np.ndarray, pct: float) -> float:
    return _safe_float(float(np.percentile(values, pct)))


def _bereken_maandkosten_componenten(waarde: float) -> tuple[float, float, float]:
    """Berekent maandkosten uitgesplitst naar (beheer, fonds, spread)."""
    if waarde <= 0:
        return (0.0, 0.0, 0.0)

    TIER_1_MAX = 100_000.0
    TIER_2_MAX = 1_000_000.0
    BEHEERKOSTEN_TIER_1 = 0.60 / 100.0
    BEHEERKOSTEN_TIER_2 = 0.50 / 100.0
    BEHEERKOSTEN_TIER_3 = 0.40 / 100.0
    FONDSKOSTEN = 0.17 / 100.0
    SPREADKOSTEN = 0.01 / 100.0

    beheerkosten_jaar = 0.0
    amount = waarde
    
    tier_1_amount = min(amount, TIER_1_MAX)
    beheerkosten_jaar += tier_1_amount * BEHEERKOSTEN_TIER_1
    amount -= tier_1_amount
    
    if amount > 0:
        tier_2_amount = min(amount, TIER_2_MAX - TIER_1_MAX)
        beheerkosten_jaar += tier_2_amount * BEHEERKOSTEN_TIER_2
        amount -= tier_2_amount
        
    if amount > 0:
        beheerkosten_jaar += amount * BEHEERKOSTEN_TIER_3

    fondskosten_jaar = waarde * FONDSKOSTEN
    spreadkosten_jaar = waarde * SPREADKOSTEN
    return (
        _safe_float(beheerkosten_jaar / 12.0),
        _safe_float(fondskosten_jaar / 12.0),
        _safe_float(spreadkosten_jaar / 12.0),
    )


def _bereken_maandkosten(waarde: float) -> float:
    """Berekent de totale kosten (beheer, fonds, spread) voor 1 maand."""
    beheer, fonds, spread = _bereken_maandkosten_componenten(waarde)
    return _safe_float(beheer + fonds + spread)


@dataclass
class _ScenarioResult:
    end_value_bruto: float
    end_value_netto: float
    realized_deposits: float
    realized_withdrawals: float
    total_costs_paid: float
    total_management_costs_paid: float
    total_fund_costs_paid: float
    total_spread_costs_paid: float
    costs_base_sum: float
    monthly_values_bruto: list[float]
    monthly_values_netto: list[float]
    monthly_net_cashflow: list[float]
    monthly_cumulative_costs: list[float]


def _simulate_single_scenario(
    *,
    inp: RekenInput,
    total_months: int,
    cashflows_by_month: dict[int, list[EenmaligeCashflow]],
    storting_start_idx: int,
    storting_end_idx: int,
    onttrekking_start_idx: int,
    onttrekking_end_idx: int,
    verwacht_rendement_by_profiel: dict[str, float],
    volatiliteit_by_profiel: dict[str, float],
    start_profiel: str,
    rng: np.random.Generator,
) -> _ScenarioResult:
    """Simulate one path using Arithmetic Returns and Shadow Accounting (Bruto/Netto)."""
    current_bruto = float(inp.startvermogen)
    current_netto = float(inp.startvermogen)
    realized_deposits = float(inp.startvermogen)
    realized_withdrawals = 0.0
    total_costs_paid = 0.0
    total_management_costs_paid = 0.0
    total_fund_costs_paid = 0.0
    total_spread_costs_paid = 0.0
    costs_base_sum = 0.0
    
    monthly_values_bruto = [current_bruto]
    monthly_values_netto = [current_netto]
    monthly_net_cashflow = [0.0]
    monthly_cumulative_costs = [0.0]

    if total_months == 0:
        return _ScenarioResult(
            end_value_bruto=current_bruto,
            end_value_netto=current_netto,
            realized_deposits=realized_deposits,
            realized_withdrawals=realized_withdrawals,
            total_costs_paid=total_costs_paid,
            total_management_costs_paid=total_management_costs_paid,
            total_fund_costs_paid=total_fund_costs_paid,
            total_spread_costs_paid=total_spread_costs_paid,
            costs_base_sum=costs_base_sum,
            monthly_values_bruto=monthly_values_bruto,
            monthly_values_netto=monthly_values_netto,
            monthly_net_cashflow=monthly_net_cashflow,
            monthly_cumulative_costs=monthly_cumulative_costs,
        )

    for month in range(1, total_months + 1):
        net_cashflow_month = 0.0

        # 1. One-time flows at BEGINNING of month
        for cashflow in cashflows_by_month.get(month, []):
            if cashflow.type == "storting":
                current_bruto += cashflow.bedrag
                current_netto += cashflow.bedrag
                realized_deposits += cashflow.bedrag
                net_cashflow_month += cashflow.bedrag
            else:
                withdrawal_bruto = min(cashflow.bedrag, current_bruto)
                withdrawal_netto = min(cashflow.bedrag, current_netto)
                current_bruto -= withdrawal_bruto
                current_netto -= withdrawal_netto
                realized_withdrawals += withdrawal_netto # Basis voor rendement is netto opname
                net_cashflow_month -= withdrawal_netto

        # 2. Profiel en rendement bepalen
        if inp.afbouw_profiel:
            profiel_maand = _profiel_with_afbouw(start_profiel, total_months, month)
        else:
            profiel_maand = inp.profiel

        annual_return = float(verwacht_rendement_by_profiel.get(profiel_maand, 0.0)) / 100.0
        annual_volatility = float(volatiliteit_by_profiel.get(profiel_maand, 0.0)) / 100.0
        
        # Arithmetische parameters voor deze maand
        mu_month = annual_return / 12.0
        sigma_month = annual_volatility / sqrt(12.0)
        r_month = rng.normal(mu_month, sigma_month)

        # 3. Rendement toepassen op beide portefeuilles
        current_bruto *= (1.0 + r_month)
        current_netto *= (1.0 + r_month)

        # 4. Kosten dynamisch afschrijven (ALLEEN op netto)
        costs_base_sum += _safe_float(current_netto)
        kosten_beheer, kosten_fonds, kosten_spread = _bereken_maandkosten_componenten(current_netto)
        kosten_deze_maand = kosten_beheer + kosten_fonds + kosten_spread
        current_netto -= kosten_deze_maand
        total_management_costs_paid += _safe_float(kosten_beheer)
        total_fund_costs_paid += _safe_float(kosten_fonds)
        total_spread_costs_paid += _safe_float(kosten_spread)
        total_costs_paid += _safe_float(kosten_deze_maand)

        # 5. Periodic flows at END of month
        if inp.periodieke_storting_maandelijks > 0 and storting_start_idx <= month <= storting_end_idx:
            current_bruto += inp.periodieke_storting_maandelijks
            current_netto += inp.periodieke_storting_maandelijks
            realized_deposits += inp.periodieke_storting_maandelijks
            net_cashflow_month += inp.periodieke_storting_maandelijks

        if inp.periodieke_onttrekking_maandelijks > 0 and onttrekking_start_idx <= month <= onttrekking_end_idx:
            withdrawal_bruto = min(inp.periodieke_onttrekking_maandelijks, current_bruto)
            withdrawal_netto = min(inp.periodieke_onttrekking_maandelijks, current_netto)
            current_bruto -= withdrawal_bruto
            current_netto -= withdrawal_netto
            realized_withdrawals += withdrawal_netto
            net_cashflow_month -= withdrawal_netto

        # 6. Floor van nul euro afdwingen (aandelen kunnen niet negatief worden)
        current_bruto = max(0.0, _safe_float(current_bruto))
        current_netto = max(0.0, _safe_float(current_netto))
        
        monthly_values_bruto.append(current_bruto)
        monthly_values_netto.append(current_netto)
        monthly_net_cashflow.append(_safe_float(net_cashflow_month))
        monthly_cumulative_costs.append(_safe_float(total_costs_paid))

    return _ScenarioResult(
        end_value_bruto=current_bruto,
        end_value_netto=current_netto,
        realized_deposits=realized_deposits,
        realized_withdrawals=realized_withdrawals,
        total_costs_paid=_safe_float(total_costs_paid),
        total_management_costs_paid=_safe_float(total_management_costs_paid),
        total_fund_costs_paid=_safe_float(total_fund_costs_paid),
        total_spread_costs_paid=_safe_float(total_spread_costs_paid),
        costs_base_sum=_safe_float(costs_base_sum),
        monthly_values_bruto=monthly_values_bruto,
        monthly_values_netto=monthly_values_netto,
        monthly_net_cashflow=monthly_net_cashflow,
        monthly_cumulative_costs=monthly_cumulative_costs,
    )


def bereken_kosten(inp: RekenInput) -> RekenOutput:
    """Calculate projections under MiFID II compliance."""
    
    # Initiele kosten schatting puur voor Jaar 1 weergave
    kosten_eur_jaar1 = _safe_float(_bereken_maandkosten(inp.startvermogen) * 12.0)
    kosten_pct_jaar1 = _safe_float((kosten_eur_jaar1 / inp.startvermogen) * 100.0) if inp.startvermogen > 0 else 0.0

    # Arithmetische brongegevens
    verwacht_rendement_by_profiel = {
        "Defensief": 2.6,
        "Matig defensief": 3.8,
        "Neutraal": 4.9,
        "Offensief": 6.0,
        "Zeer offensief": 7.2,
        "Niet beleggen": 0.0,
    }
    volatiliteit_by_profiel = {
        "Defensief": 5.0,
        "Matig defensief": 8.0,
        "Neutraal": 12.0,
        "Offensief": 16.0,
        "Zeer offensief": 20.0,
        "Niet beleggen": 0.0,
    }

    start_profiel = inp.profiel
    verwacht_rendement_pct = float(verwacht_rendement_by_profiel.get(start_profiel, 0.0))

    total_months = inp.horizon_jaren * 12
    end_date = _add_years(inp.startdatum, inp.horizon_jaren)

    cashflows_by_month: dict[int, list[EenmaligeCashflow]] = {}
    for cashflow in inp.eenmalige_cashflows:
        if cashflow.datum < inp.startdatum or cashflow.datum > end_date:
            continue
        idx = _month_index_bucket(inp.startdatum, cashflow.datum)
        if 1 <= idx <= total_months:
            cashflows_by_month.setdefault(idx, []).append(cashflow)

    storting_start_idx, storting_end_idx = _clamp_period_to_month_indices(
        startdatum=inp.startdatum,
        enddatum_horizon=end_date,
        total_months=total_months,
        periode_start=inp.periodieke_storting_startdatum,
        periode_eind=inp.periodieke_storting_einddatum,
    )
    onttrekking_start_idx, onttrekking_end_idx = _clamp_period_to_month_indices(
        startdatum=inp.startdatum,
        enddatum_horizon=end_date,
        total_months=total_months,
        periode_start=inp.periodieke_onttrekking_startdatum,
        periode_eind=inp.periodieke_onttrekking_einddatum,
    )

    rng = np.random.default_rng(seed=inp.rng_seed)
    scenario_results: list[_ScenarioResult] = []
    for _ in range(inp.n_scenarios):
        scenario_results.append(
            _simulate_single_scenario(
                inp=inp,
                total_months=total_months,
                cashflows_by_month=cashflows_by_month,
                storting_start_idx=storting_start_idx,
                storting_end_idx=storting_end_idx,
                onttrekking_start_idx=onttrekking_start_idx,
                onttrekking_end_idx=onttrekking_end_idx,
                verwacht_rendement_by_profiel=verwacht_rendement_by_profiel,
                volatiliteit_by_profiel=volatiliteit_by_profiel,
                start_profiel=start_profiel,
                rng=rng,
            )
        )

    # Aggregeren van de arrays
    end_values_bruto_arr = np.array([r.end_value_bruto for r in scenario_results], dtype=float)
    end_values_netto_arr = np.array([r.end_value_netto for r in scenario_results], dtype=float)
    
    profits_bruto_arr = np.array([r.end_value_bruto - r.realized_deposits + r.realized_withdrawals for r in scenario_results], dtype=float)
    profits_netto_arr = np.array([r.end_value_netto - r.realized_deposits + r.realized_withdrawals for r in scenario_results], dtype=float)
    
    monthly_paths_bruto_arr = np.array([r.monthly_values_bruto for r in scenario_results], dtype=float)
    monthly_paths_netto_arr = np.array([r.monthly_values_netto for r in scenario_results], dtype=float)
    monthly_net_cashflow_arr = np.array([r.monthly_net_cashflow for r in scenario_results], dtype=float)
    total_costs_paid_arr = np.array([r.total_costs_paid for r in scenario_results], dtype=float)
    total_management_costs_paid_arr = np.array([r.total_management_costs_paid for r in scenario_results], dtype=float)
    total_fund_costs_paid_arr = np.array([r.total_fund_costs_paid for r in scenario_results], dtype=float)
    total_spread_costs_paid_arr = np.array([r.total_spread_costs_paid for r in scenario_results], dtype=float)
    costs_base_sum_arr = np.array([r.costs_base_sum for r in scenario_results], dtype=float)
    monthly_cumulative_costs_arr = np.array([r.monthly_cumulative_costs for r in scenario_results], dtype=float)

    # Statistieken berekenen
    verwacht_eindvermogen_bruto = _safe_stat_mean(end_values_bruto_arr)
    verwacht_eindvermogen_netto = _safe_stat_mean(end_values_netto_arr)
    
    verwacht_eindvermogen_p10_netto = _safe_stat_percentile(end_values_netto_arr, 10)
    verwacht_eindvermogen_p50_netto = _safe_stat_percentile(end_values_netto_arr, 50)
    verwacht_eindvermogen_p90_netto = _safe_stat_percentile(end_values_netto_arr, 90)
    
    verwachte_winst_bruto = _safe_stat_mean(profits_bruto_arr)
    verwachte_winst_netto = _safe_stat_mean(profits_netto_arr)
    totale_kosten_betaald = _safe_stat_mean(total_costs_paid_arr)
    totale_kosten_impact = max(0.0, verwacht_eindvermogen_bruto - verwacht_eindvermogen_netto)
    misgelopen_rendement_op_kosten = max(0.0, totale_kosten_impact - totale_kosten_betaald)
    totale_beheerkosten_betaald = _safe_stat_mean(total_management_costs_paid_arr)
    totale_fondskosten_betaald = _safe_stat_mean(total_fund_costs_paid_arr)
    totale_spreadkosten_betaald = _safe_stat_mean(total_spread_costs_paid_arr)

    costs_base_sum = _safe_stat_mean(costs_base_sum_arr)
    if costs_base_sum > 0:
        gemiddelde_beheerkosten_pct = _safe_float((totale_beheerkosten_betaald * 12.0 / costs_base_sum) * 100.0)
        gemiddelde_fondskosten_pct = _safe_float((totale_fondskosten_betaald * 12.0 / costs_base_sum) * 100.0)
        gemiddelde_spreadkosten_pct = _safe_float((totale_spreadkosten_betaald * 12.0 / costs_base_sum) * 100.0)
    else:
        gemiddelde_beheerkosten_pct = 0.0
        gemiddelde_fondskosten_pct = 0.0
        gemiddelde_spreadkosten_pct = 0.0
    gemiddelde_totale_kosten_pct = _safe_float(
        gemiddelde_beheerkosten_pct + gemiddelde_fondskosten_pct + gemiddelde_spreadkosten_pct
    )

    # Tijdlijnen
    tijdlijn_vermogen_bruto = [_safe_float(x) for x in np.mean(monthly_paths_bruto_arr, axis=0)]
    tijdlijn_vermogen_netto = [_safe_float(x) for x in np.mean(monthly_paths_netto_arr, axis=0)]
    tijdlijn_vermogen_p10_netto = [_safe_float(x) for x in np.percentile(monthly_paths_netto_arr, 10, axis=0)]
    tijdlijn_vermogen_p90_netto = [_safe_float(x) for x in np.percentile(monthly_paths_netto_arr, 90, axis=0)]
    tijdlijn_cashflow_netto = [_safe_float(x) for x in np.mean(monthly_net_cashflow_arr, axis=0)]
    tijdlijn_kosten_cumulatief = [_safe_float(x) for x in np.mean(monthly_cumulative_costs_arr, axis=0)]

    tijdlijn_datums = [_add_months(inp.startdatum, month) for month in range(0, total_months + 1)]
    tijdlijn_profiel = [start_profiel]
    for month in range(1, total_months + 1):
        if inp.afbouw_profiel:
            tijdlijn_profiel.append(_profiel_with_afbouw(start_profiel, total_months, month))
        else:
            tijdlijn_profiel.append(inp.profiel)

    return RekenOutput(
        kosten_eur_jaar1=max(0.0, kosten_eur_jaar1),
        kosten_pct_jaar1=max(0.0, kosten_pct_jaar1),
        verwacht_rendement_pct=_safe_float(verwacht_rendement_pct),
        
        verwacht_eindvermogen_bruto=max(0.0, verwacht_eindvermogen_bruto),
        verwacht_eindvermogen_netto=max(0.0, verwacht_eindvermogen_netto),
        totale_kosten_betaald=max(0.0, totale_kosten_betaald),
        misgelopen_rendement_op_kosten=max(0.0, misgelopen_rendement_op_kosten),
        totale_kosten_impact=max(0.0, totale_kosten_impact),
        totale_beheerkosten_betaald=max(0.0, totale_beheerkosten_betaald),
        totale_fondskosten_betaald=max(0.0, totale_fondskosten_betaald),
        totale_spreadkosten_betaald=max(0.0, totale_spreadkosten_betaald),
        gemiddelde_beheerkosten_pct=max(0.0, gemiddelde_beheerkosten_pct),
        gemiddelde_fondskosten_pct=max(0.0, gemiddelde_fondskosten_pct),
        gemiddelde_spreadkosten_pct=max(0.0, gemiddelde_spreadkosten_pct),
        gemiddelde_totale_kosten_pct=max(0.0, gemiddelde_totale_kosten_pct),
        
        verwachte_winst_bruto=_safe_float(verwachte_winst_bruto),
        verwachte_winst_netto=_safe_float(verwachte_winst_netto),
        
        verwacht_eindvermogen_p10_netto=max(0.0, verwacht_eindvermogen_p10_netto),
        verwacht_eindvermogen_p50_netto=max(0.0, verwacht_eindvermogen_p50_netto),
        verwacht_eindvermogen_p90_netto=max(0.0, verwacht_eindvermogen_p90_netto),
        
        tijdlijn_datums=tijdlijn_datums,
        tijdlijn_vermogen_bruto=tijdlijn_vermogen_bruto,
        tijdlijn_vermogen_netto=tijdlijn_vermogen_netto,
        tijdlijn_vermogen_p10_netto=tijdlijn_vermogen_p10_netto,
        tijdlijn_vermogen_p90_netto=tijdlijn_vermogen_p90_netto,
        tijdlijn_profiel=tijdlijn_profiel,
        tijdlijn_cashflow_netto=tijdlijn_cashflow_netto,
        tijdlijn_kosten_cumulatief=tijdlijn_kosten_cumulatief,
    )
