# Backtest walk-forward — minrv — 9 combinaciones

Generado 2026-09-11T20:48:37Z por `backtest/walk_forward.py`.

## Parámetros

- Ventana de estimación: **rolling 24 meses** (retornos log diarios hábiles desde 2018-01-01), covarianza `ledoit` (D8).
- Rebalanceo: **quarterly** — 27 fechas de decisión, 2019-12-31 → 2026-06-30; OOS 2020-01-02 → 2026-09-08 (1635 días hábiles) (D9).
- Universo point-in-time: 16 fondos en la primera decisión, 21 en la última (entrada tras 24 m de historia; salida si deja de reportar).
- Costo de transacción: **10.0 bps** por unidad de notional transado [ESPECULATIVO]. El gate (docs/fase0_data_gate.md §4.3) documenta 0 comisiones explícitas de entrada/salida en FICs abiertos (la comisión de administración ya está descontada en el valor de unidad). 10 bps por unidad de notional transado es un proxy conservador de fricción operativa (cash drag de liquidación T+n, penalidades por pactos de permanencia) sin fuente confirmada; sensibilidad con --cost-bps.
- Turnover: one-way: 0.5 * sum(|w_target - w_drift|), w_drift = pesos previos derivados por precio. `Turnover prom.` excluye el despliegue inicial desde efectivo.
- Bandas: D6 (máximos RV / mínimos RF / máx. RV intl / máx. por fondo) como restricciones del solver; **piso RV (min_rv) por perfil: {'conservador': 0.0, 'moderado': 0.2, 'agresivo': 0.45}** [ESPECULATIVO — sensibilidad, no aprobado en D6].
- Sharpe: exceso diario sobre rf = 0.00% anual [ESPECULATIVO: rf = 0 es el estándar comparable; usa `--rf` para otra tasa], anualizado con √252.
- CVaR 95 %: media de los peores 5 % de retornos diarios (negativo = pérdida). Max DD sobre el NAV neto de costos.
- NaN: Celdas NaN intra-vida (manifest.json → gap_log) se tratan como retorno 0 tanto en la ventana de estimación como en el P&L; el retorno puenteado del primer día tras la brecha ya acumula el movimiento.

## Resumen por combinación

| Metodología | Perfil | Ret. anual | Vol. anual | Sharpe | Max DD | CVaR 95 % (diario) | Turnover prom. | Costo total | N rebal. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| markowitz | conservador | 7.23% | 0.97% | 7.18 | -2.94% | -0.11% | 6.05% | 0.41% | 27 |
| markowitz | moderado | 7.08% | 2.85% | 2.41 | -10.07% | -0.38% | 17.11% | 0.99% | 27 |
| markowitz | agresivo | 7.83% | 6.20% | 1.25 | -17.78% | -0.90% | 22.26% | 1.26% | 27 |
| risk_parity | conservador | 7.82% | 1.62% | 4.65 | -4.29% | -0.21% | 2.87% | 0.25% | 27 |
| risk_parity | moderado | 7.47% | 3.09% | 2.35 | -10.28% | -0.42% | 3.17% | 0.26% | 27 |
| risk_parity | agresivo | 7.41% | 6.37% | 1.15 | -19.28% | -0.91% | 3.93% | 0.30% | 27 |
| hrp | conservador | 7.52% | 1.02% | 7.12 | -2.97% | -0.11% | 5.71% | 0.40% | 27 |
| hrp | moderado | 7.52% | 3.28% | 2.23 | -10.19% | -0.46% | 6.02% | 0.41% | 27 |
| hrp | agresivo | 7.61% | 7.08% | 1.07 | -19.78% | -1.03% | 6.03% | 0.41% | 27 |

## Validación ex-post de volatilidad (gate §3.2)

Banda esperada por perfil ±2%; fuera de banda ⇒ el gate pide revisar los límites (D6).

| Metodología | Perfil | Vol. realizada | Banda esperada | Veredicto |
|---|---|---:|---:|---|
| markowitz | conservador | 0.97% | 3.00%–5.00% | **FUERA** |
| markowitz | moderado | 2.85% | 7.00%–10.00% | **FUERA** |
| markowitz | agresivo | 6.20% | 11.00%–15.00% | **FUERA** |
| risk_parity | conservador | 1.62% | 3.00%–5.00% | dentro |
| risk_parity | moderado | 3.09% | 7.00%–10.00% | **FUERA** |
| risk_parity | agresivo | 6.37% | 11.00%–15.00% | **FUERA** |
| hrp | conservador | 1.02% | 3.00%–5.00% | dentro |
| hrp | moderado | 3.28% | 7.00%–10.00% | **FUERA** |
| hrp | agresivo | 7.08% | 11.00%–15.00% | **FUERA** |

## Corridas de sensibilidad disponibles

- `results/backtest_summary_cost0.md`
- `results/backtest_summary_cost25.md`
- `results/backtest_summary_expanding.md`
- `results/backtest_summary_monthly.md`

## Fondos activos por fecha de decisión

| Fecha | N activos |
|---|---:|
| 2019-12-31 | 16 |
| 2020-03-31 | 17 |
| 2020-06-30 | 17 |
| 2020-09-30 | 17 |
| 2020-12-31 | 18 |
| 2021-03-31 | 19 |
| 2021-06-30 | 19 |
| 2021-09-30 | 19 |
| 2021-12-31 | 19 |
| 2022-03-31 | 19 |
| 2022-06-30 | 19 |
| 2022-09-30 | 20 |
| 2022-12-30 | 20 |
| 2023-03-31 | 21 |
| 2023-06-30 | 21 |
| 2023-09-29 | 21 |
| 2023-12-29 | 21 |
| 2024-03-27 | 21 |
| 2024-06-28 | 21 |
| 2024-09-30 | 21 |
| 2024-12-31 | 21 |
| 2025-03-31 | 21 |
| 2025-06-27 | 21 |
| 2025-09-30 | 21 |
| 2025-12-31 | 21 |
| 2026-03-31 | 21 |
| 2026-06-30 | 21 |
