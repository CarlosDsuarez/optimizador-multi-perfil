# Backtest walk-forward — monthly — 9 combinaciones

Generado 2026-09-11T20:48:13Z por `backtest/walk_forward.py`.

## Parámetros

- Ventana de estimación: **rolling 24 meses** (retornos log diarios hábiles desde 2018-01-01), covarianza `ledoit` (D8).
- Rebalanceo: **monthly** — 81 fechas de decisión, 2019-12-31 → 2026-08-31; OOS 2020-01-02 → 2026-09-08 (1635 días hábiles) (D9).
- Universo point-in-time: 16 fondos en la primera decisión, 21 en la última (entrada tras 24 m de historia; salida si deja de reportar).
- Costo de transacción: **10.0 bps** por unidad de notional transado [ESPECULATIVO]. El gate (docs/fase0_data_gate.md §4.3) documenta 0 comisiones explícitas de entrada/salida en FICs abiertos (la comisión de administración ya está descontada en el valor de unidad). 10 bps por unidad de notional transado es un proxy conservador de fricción operativa (cash drag de liquidación T+n, penalidades por pactos de permanencia) sin fuente confirmada; sensibilidad con --cost-bps.
- Turnover: one-way: 0.5 * sum(|w_target - w_drift|), w_drift = pesos previos derivados por precio. `Turnover prom.` excluye el despliegue inicial desde efectivo.
- Bandas: D6 (máximos RV / mínimos RF / máx. RV intl / máx. por fondo) como restricciones del solver; sin piso de RV (min_rv = 0).
- Sharpe: exceso diario sobre rf = 0.00% anual [ESPECULATIVO: rf = 0 es el estándar comparable; usa `--rf` para otra tasa], anualizado con √252.
- CVaR 95 %: media de los peores 5 % de retornos diarios (negativo = pérdida). Max DD sobre el NAV neto de costos.
- NaN: Celdas NaN intra-vida (manifest.json → gap_log) se tratan como retorno 0 tanto en la ventana de estimación como en el P&L; el retorno puenteado del primer día tras la brecha ya acumula el movimiento.

## Resumen por combinación

| Metodología | Perfil | Ret. anual | Vol. anual | Sharpe | Max DD | CVaR 95 % (diario) | Turnover prom. | Costo total | N rebal. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| markowitz | conservador | 7.29% | 0.92% | 7.62 | -2.55% | -0.11% | 2.91% | 0.57% | 81 |
| markowitz | moderado | 7.34% | 0.97% | 7.30 | -2.55% | -0.12% | 2.49% | 0.50% | 81 |
| markowitz | agresivo | 7.34% | 0.97% | 7.30 | -2.55% | -0.12% | 2.49% | 0.50% | 81 |
| risk_parity | conservador | 7.83% | 1.55% | 4.87 | -3.85% | -0.20% | 1.18% | 0.29% | 81 |
| risk_parity | moderado | 7.83% | 1.55% | 4.87 | -3.85% | -0.20% | 1.18% | 0.29% | 81 |
| risk_parity | agresivo | 7.83% | 1.55% | 4.87 | -3.85% | -0.20% | 1.18% | 0.29% | 81 |
| hrp | conservador | 7.45% | 0.96% | 7.51 | -2.55% | -0.10% | 2.93% | 0.57% | 81 |
| hrp | moderado | 7.54% | 0.98% | 7.45 | -2.55% | -0.10% | 2.78% | 0.54% | 81 |
| hrp | agresivo | 7.54% | 0.98% | 7.45 | -2.55% | -0.10% | 2.78% | 0.54% | 81 |

## Validación ex-post de volatilidad (gate §3.2)

Banda esperada por perfil ±2%; fuera de banda ⇒ el gate pide revisar los límites (D6).

| Metodología | Perfil | Vol. realizada | Banda esperada | Veredicto |
|---|---|---:|---:|---|
| markowitz | conservador | 0.92% | 3.00%–5.00% | **FUERA** |
| markowitz | moderado | 0.97% | 7.00%–10.00% | **FUERA** |
| markowitz | agresivo | 0.97% | 11.00%–15.00% | **FUERA** |
| risk_parity | conservador | 1.55% | 3.00%–5.00% | dentro |
| risk_parity | moderado | 1.55% | 7.00%–10.00% | **FUERA** |
| risk_parity | agresivo | 1.55% | 11.00%–15.00% | **FUERA** |
| hrp | conservador | 0.96% | 3.00%–5.00% | **FUERA** |
| hrp | moderado | 0.98% | 7.00%–10.00% | **FUERA** |
| hrp | agresivo | 0.98% | 11.00%–15.00% | **FUERA** |

## Corridas de sensibilidad disponibles

- `results/backtest_summary_cost0.md`
- `results/backtest_summary_cost25.md`
- `results/backtest_summary_expanding.md`
- `results/backtest_summary_minrv.md`

## Fondos activos por fecha de decisión

| Fecha | N activos |
|---|---:|
| 2019-12-31 | 16 |
| 2020-01-31 | 16 |
| 2020-02-28 | 17 |
| 2020-03-31 | 17 |
| 2020-04-30 | 17 |
| 2020-05-29 | 17 |
| 2020-06-30 | 17 |
| 2020-07-31 | 17 |
| 2020-08-31 | 17 |
| 2020-09-30 | 17 |
| 2020-10-30 | 17 |
| 2020-11-30 | 18 |
| 2020-12-31 | 18 |
| 2021-01-29 | 18 |
| 2021-02-26 | 19 |
| 2021-03-31 | 19 |
| 2021-04-30 | 19 |
| 2021-05-31 | 19 |
| 2021-06-30 | 19 |
| 2021-07-30 | 19 |
| 2021-08-31 | 19 |
| 2021-09-30 | 19 |
| 2021-10-29 | 19 |
| 2021-11-30 | 19 |
| 2021-12-31 | 19 |
| 2022-01-31 | 19 |
| 2022-02-28 | 19 |
| 2022-03-31 | 19 |
| 2022-04-29 | 19 |
| 2022-05-31 | 19 |
| 2022-06-30 | 19 |
| 2022-07-29 | 20 |
| 2022-08-31 | 20 |
| 2022-09-30 | 20 |
| 2022-10-31 | 20 |
| 2022-11-30 | 20 |
| 2022-12-30 | 20 |
| 2023-01-31 | 20 |
| 2023-02-28 | 20 |
| 2023-03-31 | 21 |
| 2023-04-28 | 21 |
| 2023-05-31 | 21 |
| 2023-06-30 | 21 |
| 2023-07-31 | 21 |
| 2023-08-31 | 21 |
| 2023-09-29 | 21 |
| 2023-10-31 | 21 |
| 2023-11-30 | 21 |
| 2023-12-29 | 21 |
| 2024-01-31 | 21 |
| 2024-02-29 | 21 |
| 2024-03-27 | 21 |
| 2024-04-30 | 21 |
| 2024-05-31 | 21 |
| 2024-06-28 | 21 |
| 2024-07-31 | 21 |
| 2024-08-30 | 21 |
| 2024-09-30 | 21 |
| 2024-10-31 | 21 |
| 2024-11-29 | 21 |
| 2024-12-31 | 21 |
| 2025-01-31 | 21 |
| 2025-02-28 | 21 |
| 2025-03-31 | 21 |
| 2025-04-30 | 21 |
| 2025-05-30 | 21 |
| 2025-06-27 | 21 |
| 2025-07-31 | 21 |
| 2025-08-29 | 21 |
| 2025-09-30 | 21 |
| 2025-10-31 | 21 |
| 2025-11-28 | 21 |
| 2025-12-31 | 21 |
| 2026-01-30 | 21 |
| 2026-02-27 | 21 |
| 2026-03-31 | 21 |
| 2026-04-30 | 21 |
| 2026-05-29 | 21 |
| 2026-06-30 | 21 |
| 2026-07-31 | 21 |
| 2026-08-31 | 21 |
