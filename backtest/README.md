# `backtest/` — walk-forward sin look-ahead (Fase 3)

| Ruta | Qué es |
|---|---|
| `walk_forward.py` | Motor: `rebalance_dates` → `active_universe` (point-in-time) → `estimation_window` (solo datos ≤ t) → optimizador inyectado → `simulate_period` (buy-and-hold, costo el primer día) → `compute_metrics` → `write_outputs` + `manifest.json → backtest[_tag]`. |
| `results/backtest_summary[_tag].parquet` | 9 filas (metodología × perfil): métricas, parámetros y columna anidada `weights` [date, fund_id, weight]. |
| `results/backtest_weights[_tag].parquet` | Formato largo: pesos por fecha de decisión y fondo, `active`, `turnover`, `cost`, `n_active`. |
| `results/backtest_daily[_tag].parquet` | Retorno diario neto y NAV por combinación. |
| `results/backtest_summary[_tag].md` | Tabla resumen + validación ex-post de volatilidad (gate §3.2) + fondos activos por fecha. |

Parámetros aprobados (D8/D9): rolling 24 m, retornos log diarios hábiles desde 2018-01-01, Ledoit-Wolf,
rebalanceo trimestral, primera decisión 2019-12-31 (datos ≤ esa fecha), pesos vivos desde 2020-01-02.
Supuestos [ESPECULATIVO] parametrizables: `--cost-bps 10` (gate: 0 comisiones explícitas; proxy de fricción),
`--rf 0`, `--min-rv` (piso de RV por perfil; base = sin piso, ver validación ex-post en el md).

```bash
.venv/bin/python -m backtest.walk_forward                              # base → results/backtest_summary.*
.venv/bin/python -m backtest.walk_forward --rebalance monthly --tag monthly
.venv/bin/python -m backtest.walk_forward --window-kind expanding --tag expanding
.venv/bin/python -m backtest.walk_forward --cost-bps 25 --tag cost25
.venv/bin/python -m backtest.walk_forward --min-rv 0 0.20 0.45 --tag minrv   # pisos Art. 2.6.12.1.4 (sensibilidad)
.venv/bin/python -m pytest tests/test_walk_forward.py --cov=backtest.walk_forward --cov-fail-under=85
```

Exit codes: `0` OK · `2` `BacktestError` (universo insuficiente, salida del optimizador no es un portafolio,
invariante de look-ahead violada).

Garantías anti-look-ahead: la ventana en `t` es `(t − 24m, t]` (rolling) y el motor aborta si
`window.index.max() > t`; el P&L de cada periodo usa solo filas `> t`. `tests/test_walk_forward.py` inyecta
un outlier futuro y exige pesos y retornos **bit a bit idénticos** antes de esa fecha.
