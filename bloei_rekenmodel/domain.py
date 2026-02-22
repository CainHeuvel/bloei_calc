"""Domain models for the Bloei Rekenmodule calculation engine."""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional


_TOEGESTANE_CASHFLOW_TYPES = {"storting", "onttrekking"}
_TOEGESTANE_PROFIELEN = {
    "Defensief",
    "Matig defensief",
    "Neutraal",
    "Offensief",
    "Zeer offensief",
    "Niet beleggen",
}


@dataclass
class EenmaligeCashflow:
    """Represents a one-time cashflow event.

    Semantics:
    - bedrag is always non-negative.
    - direction is defined by ``type`` only.
    - valid type values are ``storting`` and ``onttrekking``.
    """

    bedrag: float
    datum: date
    type: str

    def __post_init__(self) -> None:
        if self.bedrag < 0:
            raise ValueError("EenmaligeCashflow.bedrag moet >= 0 zijn.")
        if self.type not in _TOEGESTANE_CASHFLOW_TYPES:
            raise ValueError(
                f"EenmaligeCashflow.type moet een van {_TOEGESTANE_CASHFLOW_TYPES} zijn."
            )


@dataclass
class RekenInput:
    """Input parameters for the projection engine."""

    startvermogen: float
    profiel: str
    startdatum: date
    horizon_jaren: int
    n_scenarios: int
    periodieke_storting_maandelijks: float = 0.0
    periodieke_onttrekking_maandelijks: float = 0.0
    periodieke_storting_startdatum: Optional[date] = None
    periodieke_storting_einddatum: Optional[date] = None
    periodieke_onttrekking_startdatum: Optional[date] = None
    periodieke_onttrekking_einddatum: Optional[date] = None
    eenmalige_cashflows: Optional[List[EenmaligeCashflow]] = None
    afbouw_profiel: bool = False
    rng_seed: int = 42

    def __post_init__(self) -> None:
        if self.eenmalige_cashflows is None:
            self.eenmalige_cashflows = []

        self.horizon_jaren = int(self.horizon_jaren)
        self.n_scenarios = int(self.n_scenarios)
        self.rng_seed = int(self.rng_seed)

        if self.startvermogen < 0:
            raise ValueError("startvermogen moet >= 0 zijn.")
        if self.horizon_jaren < 0:
            raise ValueError("horizon_jaren moet >= 0 zijn.")
        if self.n_scenarios < 1:
            raise ValueError("n_scenarios moet >= 1 zijn.")
        if self.periodieke_storting_maandelijks < 0:
            raise ValueError("periodieke_storting_maandelijks moet >= 0 zijn.")
        if self.periodieke_onttrekking_maandelijks < 0:
            raise ValueError("periodieke_onttrekking_maandelijks moet >= 0 zijn.")
        if self.profiel not in _TOEGESTANE_PROFIELEN:
            raise ValueError(f"profiel moet een van {_TOEGESTANE_PROFIELEN} zijn.")

        for cashflow in self.eenmalige_cashflows:
            if not isinstance(cashflow, EenmaligeCashflow):
                raise ValueError(
                    "eenmalige_cashflows mag alleen EenmaligeCashflow-objecten bevatten."
                )


@dataclass
class RekenOutput:
    """Output results from the projection engine.

    Important timing semantics used consistently in engine + UI:
    - ``tijdlijn_datums[m]`` is the month-step marker ``startdatum + m maanden``.
      It marks the end of simulation step m.
    - One-time cashflows are applied at the BEGINNING of a calendar month bucket.
    - Periodic cashflows are applied at the END of each month.
    """

    kosten_eur_jaar1: float
    kosten_pct_jaar1: float
    verwacht_rendement_pct: float
    
    # MiFID II uitsplitsing
    verwacht_eindvermogen_bruto: float
    verwacht_eindvermogen_netto: float
    totale_kosten_betaald: float
    misgelopen_rendement_op_kosten: float
    totale_kosten_impact: float
    totale_beheerkosten_betaald: float
    totale_fondskosten_betaald: float
    totale_spreadkosten_betaald: float
    gemiddelde_beheerkosten_pct: float
    gemiddelde_fondskosten_pct: float
    gemiddelde_spreadkosten_pct: float
    gemiddelde_totale_kosten_pct: float
    
    verwachte_winst_bruto: float
    verwachte_winst_netto: float
    
    verwacht_eindvermogen_p10_netto: float
    verwacht_eindvermogen_p50_netto: float
    verwacht_eindvermogen_p90_netto: float
    
    tijdlijn_datums: List[date]
    tijdlijn_vermogen_bruto: List[float]
    tijdlijn_vermogen_netto: List[float]
    tijdlijn_vermogen_p10_netto: List[float]
    tijdlijn_vermogen_p90_netto: List[float]
    tijdlijn_profiel: List[str]
    tijdlijn_cashflow_netto: List[float]
    tijdlijn_kosten_cumulatief: List[float]
