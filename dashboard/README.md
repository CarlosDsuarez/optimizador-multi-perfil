# `dashboard/` — dendrograma HRP interactivo ↔ AG Grid (Fase 4)

| Ruta | Qué es |
|---|---|
| `app.py` | `create_app()` (factoría Dash), layout dbc, callbacks `asof_choices` / `compute` / `render_grid` / `render_figure` / `sync` |
| `data.py` | carga de `data/cleaned/*`, `fingerprint`, fechas as-of, `window_slice` (reusa `active_universe` / `estimation_window` del backtest) |
| `cache.py` | Flask-Caching: `distance_bundle` (corr → distancia → linkage → hojas → clusters → Σ LW) y `portfolio_bundle` (`optimize`) |
| `figures.py` | dendrograma horizontal desde la matriz de linkage; cada enlace lleva `customdata` = fondos de su subárbol |
| `grid.py` | filas del AG Grid: peso, peso HRP pre-QP, contribución a varianza (fondo y cluster), métricas por fondo |
| `assets/dashboard.css` | color de fila seleccionada (AG Grid Theming API) |

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m dashboard.app                 # http://127.0.0.1:8050
.venv/bin/python -m pytest tests/test_dashboard.py --cov=dashboard --cov-fail-under=85
```

**Interacción.** Hover sobre una hoja resalta su fila; sobre un enlace (U) resalta todo el subárbol; click
fija/libera la selección (persiste al retirar el cursor). Cambiar perfil/metodología/ventana recalcula sin
recargar. El dendrograma no depende de la metodología (solo de la matriz de distancia), así que se muestra
también con Markowitz y Risk Parity; con HRP el grid añade la columna de pesos antes de la proyección QP (D5).

**Cache.** `distance_bundle` se memoiza con clave `(universo, inicio, fin, fingerprint)` — perfil y metodología
no están en la clave, por lo que cambiar de perfil nunca recalcula la matriz de distancia
(`tests/test_dashboard.py::test_profile_and_method_changes_hit_distance_cache`). `fingerprint` = sha256 del
parquet. `returns`/`fingerprint` se calculan una sola vez en `load_data` al arrancar el proceso: un re-ingest
en caliente no invalida nada mientras el proceso sigue corriendo. Lo que sí logra el fingerprint es que, tras
reiniciar el proceso después de un re-ingest, un backend persistente (`FileSystemCache`) nunca sirva un
bundle calculado con el parquet anterior. TTL `DIST_TTL_SECONDS = 24 h` (cadencia diaria SFC; solo acota
memoria, ver comentario en `cache.py`). Backend `SimpleCache` por defecto; para varios workers:
`DASH_CACHE_TYPE=FileSystemCache DASH_CACHE_DIR=.cache/dashboard`.

**Tests.** Los callbacks se prueban por `POST /_dash-update-component` con el test client de Flask (ruta real de
Dash, sin navegador). El test E2E con `dash_duo` se omite si no hay `chromedriver` en PATH (`dash[testing]`
pinea `selenium<=4.2.0`, sin Selenium Manager): `brew install chromedriver` para ejecutarlo.

Fuera de alcance (decisión de Fase 4): exportación PDF / tear sheet, estado en URL, sincronización grid → dendrograma.
