"""Minimal regression tests for Bloei rekenmodel semantics."""

import unittest
from datetime import date

from bloei_rekenmodel import EenmaligeCashflow, RekenInput, bereken_kosten


class RekenmodelTests(unittest.TestCase):
    def test_negative_cashflow_amount_raises(self) -> None:
        with self.assertRaises(ValueError):
            EenmaligeCashflow(bedrag=-1.0, datum=date(2026, 1, 1), type="storting")

    def test_invalid_cashflow_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            EenmaligeCashflow(bedrag=100.0, datum=date(2026, 1, 1), type="foo")

    def test_cashflow_month_bucketing_ignores_day_of_month(self) -> None:
        base_kwargs = dict(
            startvermogen=1000.0,
            profiel="Niet beleggen",
            startdatum=date(2026, 1, 15),
            horizon_jaren=1,
            n_scenarios=1,
            rng_seed=42,
        )
        inp_day_1 = RekenInput(
            **base_kwargs,
            eenmalige_cashflows=[EenmaligeCashflow(200.0, date(2026, 3, 1), "storting")],
        )
        inp_day_30 = RekenInput(
            **base_kwargs,
            eenmalige_cashflows=[EenmaligeCashflow(200.0, date(2026, 3, 30), "storting")],
        )

        out_day_1 = bereken_kosten(inp_day_1)
        out_day_30 = bereken_kosten(inp_day_30)
        self.assertEqual(out_day_1.verwacht_eindvermogen_netto, out_day_30.verwacht_eindvermogen_netto)
        self.assertEqual(out_day_1.tijdlijn_cashflow_netto, out_day_30.tijdlijn_cashflow_netto)

    def test_reproducible_with_fixed_seed(self) -> None:
        inp = RekenInput(
            startvermogen=100000.0,
            profiel="Neutraal",
            startdatum=date(2026, 1, 1),
            horizon_jaren=5,
            n_scenarios=200,
            rng_seed=42,
        )
        out_1 = bereken_kosten(inp)
        out_2 = bereken_kosten(inp)
        self.assertEqual(out_1.verwacht_eindvermogen_netto, out_2.verwacht_eindvermogen_netto)
        self.assertEqual(out_1.tijdlijn_vermogen_netto, out_2.tijdlijn_vermogen_netto)

    def test_horizon_zero_returns_single_timeline_point(self) -> None:
        inp = RekenInput(
            startvermogen=12345.0,
            profiel="Defensief",
            startdatum=date(2026, 2, 20),
            horizon_jaren=0,
            n_scenarios=1,
        )
        out = bereken_kosten(inp)
        self.assertEqual(len(out.tijdlijn_datums), 1)
        self.assertEqual(len(out.tijdlijn_vermogen_netto), 1)
        self.assertEqual(out.tijdlijn_vermogen_netto[0], 12345.0)
        self.assertEqual(out.verwacht_eindvermogen_netto, 12345.0)
        self.assertTrue(out.verwachte_winst_netto == 0.0)

    def test_withdrawal_clamping_prevents_overstated_profit(self) -> None:
        inp = RekenInput(
            startvermogen=100.0,
            profiel="Niet beleggen",
            startdatum=date(2026, 1, 1),
            horizon_jaren=1,
            n_scenarios=1,
            periodieke_onttrekking_maandelijks=1000.0,
        )
        out = bereken_kosten(inp)
        self.assertEqual(out.verwacht_eindvermogen_netto, 0.0)
        self.assertEqual(out.verwachte_winst_bruto, 0.0)
        self.assertEqual(out.tijdlijn_cashflow_netto[1], -99.935)
        self.assertTrue(all(v >= -100.0 for v in out.tijdlijn_cashflow_netto))
        total_realized_withdrawals = sum(-v for v in out.tijdlijn_cashflow_netto if v < 0)
        self.assertEqual(total_realized_withdrawals, 99.935)


if __name__ == "__main__":
    unittest.main()
