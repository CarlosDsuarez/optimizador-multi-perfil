# `optimization/` — capa matemática (Fase 2)

`portfolio_optimizer.optimize(returns, method, profile, constraints=None, *, categories, rf=0.0,
periods_per_year=252, cov_method="ledoit") -> OptimizationResult`

| Método | Implementación | Bandas |
|---|---|---|
| `markowitz` | PyPortfolioOpt 1.6.0 `EfficientFrontier.max_sharpe` (CLARABEL, tol 1e-10) | bounds + `A w ≤ b` en el solver |
| `risk_parity` | Riskfolio-Lib 7.3.0 `Portfolio.rp_optimization` | `ainequality/binequality` (`A w ≤ b`) |
| `hrp` | Riskfolio-Lib 7.3.0 `HCPortfolio.optimization(model="HRP")` + proyección QP exacta (D5) | `min ‖w − w_hrp‖²` s.a. bandas |

Bandas D6/D7 en `PROFILE_CONSTRAINTS` (máx. RV combinada / mín. RF / máx. RV intl / máx. por fondo; look-through
50 % de mixtos). `min_rv` (piso de RV) es opcional y vale 0 en el caso base. `OptimizationResult` conserva
`linkage_matrix`, `leaf_order`, `cluster_labels` y `raw_weights` para el dendrograma de HRP (Fase 4).
Firmas verificadas contra la versión instalada: etiquetas [VERIFICADO] en los docstrings de `_solve_*`.

```bash
.venv/bin/python -m pytest tests/test_portfolio_optimizer.py --cov=optimization.portfolio_optimizer --cov-fail-under=95
```
