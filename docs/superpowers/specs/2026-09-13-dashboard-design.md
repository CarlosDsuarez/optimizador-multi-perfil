# Fase 4 — Dashboard interactivo (dendrograma HRP ↔ AG Grid)

**Estado:** diseño aprobado en chat 2026-09-13 (Carlos). Decisiones D1–D11 del gate (`docs/fase0_data_gate.md`)
y las bandas D6 de `optimization/portfolio_optimizer.py` se toman como dadas.

## 1. Objetivo

Aplicación Dash que, para una combinación perfil × metodología × ventana, muestra a la izquierda el
dendrograma de la jerarquía HRP y a la derecha una tabla (AG Grid) con pesos, contribución al riesgo y
métricas por fondo. **Requisito funcional central:** pasar el cursor sobre una hoja (o un enlace) del
dendrograma resalta la(s) fila(s) correspondiente(s) en la tabla, sin recargar la página. Cambiar perfil o
metodología recalcula (o lee de cache) sin recargar.

Fuera de alcance (decisión explícita): exportación PDF/tear sheet, persistencia de estado en URL,
sincronización inversa grid → dendrograma.

## 2. Stack (pines exactos, verificados con `pip install --dry-run`)

| Paquete | Versión | Notas |
|---|---|---|
| `dash` | 4.4.1 | `app.run()`; callbacks sin estado; `dcc.Store` para estado de cliente |
| `dash-ag-grid` | 35.3.0 | requiere `dash>=2` |
| `flask-caching` | 2.5.1 | `Cache(app.server, config=...)` |
| `dash-bootstrap-components` | 2.0.4 | requiere `dash>=3.0.4`; layout |
| `dash[testing]` | (extra) | pinea `selenium<=4.2.0` → sin Selenium Manager; requiere `chromedriver` en PATH |

`plotly 6.8.0` ya instalado. Todo se instala en `.venv/` y se pinea en `requirements.txt`.

## 3. Módulos

```
dashboard/
  __init__.py
  app.py       create_app(returns=None, funds_master=None, cache_config=None) -> dash.Dash
  data.py      load_returns(), load_funds_master(), asof_options(), window_slice()
  cache.py     cache = Cache(); distance_bundle(); portfolio_bundle(); fingerprint()
  figures.py   build_dendrogram(bundle, cluster_labels, labels, highlight) -> go.Figure
  grid.py      COLUMN_DEFS, build_rows(result, bundle, window) -> list[dict]
  README.md
tests/test_dashboard.py
```

`create_app` es una factoría: los tests inyectan retornos sintéticos y `CACHE_TYPE=SimpleCache`. No hay
variables globales mutables de sesión; el único estado de proceso es la cache de Flask-Caching (atada a
`app.server`) y los datos inmutables cargados al arrancar. `app.run()` solo bajo `__main__`.

## 4. Datos y ventana

- Retornos: `data/cleaned/returns_matrix.parquet` (log diarios, 2016-01 → hoy, 21 fondos).
- Controles: perfil (`PROFILES`), metodología (`METHODS`), fecha as-of, lookback (12/24/36 meses, default 24 = D8).
- As-of options = `backtest.walk_forward.rebalance_dates(index, BacktestConfig())` + última fecha del índice
  (default = última fecha).
- Universo point-in-time = `active_universe(returns, asof, BacktestConfig(window_months=lookback))`.
- Ventana = `estimation_window(returns[universe], asof, BacktestConfig(window_months=lookback))` →
  `(asof − lookback, asof]`, soporte común, NaN intra-vida → 0 (misma política que el backtest).
- `fingerprint` = 16 hex de sha256 del parquet (o del contenido del DataFrame inyectado). Un re-ingest cambia
  la clave y por tanto invalida la cache sin reiniciar el proceso.

## 5. Cache (no negociable)

```python
@cache.memoize(timeout=DIST_TTL_SECONDS)
def distance_bundle(universe: tuple[str, ...], start: str, end: str, fingerprint: str) -> DistanceBundle
```

- Calcula: `corr = window.corr()` (Pearson), `dist = sqrt(clip((1 − corr)/2, 0, 1))`,
  `linkage = scipy.cluster.hierarchy.linkage(squareform(dist, checks=False), 'single', optimal_ordering=True)`,
  `leaf_order = leaves_list(linkage)`. Réplica exacta de Riskfolio-Lib 7.3.0 `HCPortfolio._hierarchical_clustering`
  (líneas 360, 51, 411 de la fuente instalada) → `linkage` debe ser `allclose` a
  `OptimizationResult.linkage_matrix` (test).
- Clave = hash de los argumentos (Flask-Caching memoize): universo (tupla ordenada), fechas de la ventana,
  fingerprint. Perfil y metodología **no** forman parte de la clave.
- `DIST_TTL_SECONDS = 24 * 3600`. Justificación (documentada en código): los datos SFC se publican con
  cadencia diaria, y el fingerprint ya invalida cualquier cambio real de datos; el TTL solo acota la memoria
  del backend y asegura que una entrada huérfana (fingerprint viejo) desaparezca en un día.
- `portfolio_bundle(universe, start, end, fingerprint, method, profile)` también memoizada (mismo TTL): envuelve
  `optimize()`; guarda pesos, raw_weights, cluster_labels, exposiciones y métricas. Cambiar perfil → si ya
  visitado lee cache, si no llama `optimize()` (~1 s) **pero nunca recalcula la matriz de distancia**.
- Backend: `CACHE_TYPE=SimpleCache` por defecto (`CACHE_THRESHOLD=500`). Variables de entorno
  `DASH_CACHE_TYPE=FileSystemCache` + `DASH_CACHE_DIR` para despliegue multi-worker (gunicorn).
- Instrumentación: `distance_bundle` delega en `_compute_distance(window)` (función pura, monkeypatcheable)
  para contar llamadas en tests.

## 6. Layout y callbacks

Layout (dbc): fila superior de controles (`dcc.Dropdown` × 4); debajo `dbc.Row` con dos columnas:
izquierda `dcc.Graph(id="dendro", clear_on_unhover=True)`, derecha badges (exp. RV / RF / RV intl, vol,
Sharpe, n fondos) + `dag.AgGrid(id="grid", getRowId="params.data.fund_id", rowSelection multiRow sin
checkboxes, sortable/filterable)`. `dbc.Alert(id="error")` oculto salvo error.

Stores: `dcc.Store(id="result")` (payload JSON: rows, linkage, leaf_order, cluster_labels, fund_ids,
labels, badges, método/perfil/ventana); `dcc.Store(id="pinned")` (lista de fund_ids fijados por click).

| Callback | Inputs / State | Outputs | Comportamiento |
|---|---|---|---|
| `compute` | perfil, método, asof, lookback | `result.data`, `error.children`, `error.is_open` | `distance_bundle` + `portfolio_bundle`; `OptimizationError` / universo < 2 → alerta, `result` = `no_update` |
| `render` | `result.data`, `pinned.data` | `dendro.figure`, `grid.rowData`, badges | filas ordenadas por `leaf_order`; figura con pins resaltados |
| `sync` | `dendro.hoverData`, `dendro.clickData`, State `result.data`, `pinned.data` | `grid.selectedRows`, `grid.scrollTo`, `pinned.data` | hover hoja → `{"ids":[fund]}`; hover enlace → ids del subárbol; click alterna pin; hover-out sin pin → `{"ids": []}`, con pin → ids fijados |

`sync` usa `dash.ctx.triggered_id` / `triggered_prop_ids` para distinguir hover, click y hover-out. Todas las
funciones de callback son puras respecto de sus argumentos (testables vía `POST /_dash-update-component`).

## 7. Dendrograma (`figures.py`)

`plotly.figure_factory.create_dendrogram` se evaluó y se descartó: sus trazas no exponen la relación
nodo → hojas (necesaria para el hover de subárbol) y solo colorean por `color_threshold`. Se usa la misma
geometría base:

1. `d = scipy.cluster.hierarchy.dendrogram(Z, no_plot=True, link_color_func=str)` → `icoord`, `dcoord`,
   `leaves` y `color_list` = ids de nodo en el **mismo orden** que los enlaces (scipy los anexa en la misma
   recursión).
2. Nodo id `k ≥ n` → hojas via `to_tree(Z, rd=True)[1][k].pre_order()`.
3. Una traza `go.Scatter` por enlace (forma U, `mode="lines"`), `customdata` = lista de `fund_id` del subárbol,
   `hovertemplate` con nº de fondos y distancia; color = color del cluster si todas sus hojas comparten
   `cluster_labels`, gris si el enlace une clusters distintos.
4. Traza de marcadores en las hojas (`customdata=[fund_id]`, `hovertemplate` nombre + categoría + peso).
5. Orientación horizontal (`icoord` → eje y, `dcoord` → eje x); tick labels = etiqueta corta del fondo en el
   orden de `leaves` (== `leaf_order`, así la tabla alineada por defecto lee igual que las hojas).
6. Resaltado: enlaces/hojas cuyo subárbol ⊆ pins → `line.width` mayor / marcador mayor.

## 8. Grid (`grid.py`)

Columnas: fondo (nombre corto) · categoría · cluster · peso % · peso raw HRP % (solo `method == "hrp"`,
columna oculta en otro caso) · contribución a varianza % · contribución a varianza del cluster % · retorno
anual · vol anual · Sharpe (rf 0) · max drawdown (ventana). `valueFormatter` a 2 decimales; `rowClassRules`
resalta filas seleccionadas; `defaultColDef = {sortable, filter, resizable}`.

Contribución a varianza: `RC_i = w_i (Σ w)_i / (wᵀ Σ w)` con Σ Ledoit-Wolf anualizada de
`optimization.portfolio_optimizer._estimate(window, "ledoit", 252)` (la misma del optimizador). Contribución
del cluster = suma de `RC_i` sobre las hojas del cluster de `i`. Métricas por fondo sobre la ventana:
`mean*252`, `std*sqrt(252)`, Sharpe = ratio, max DD sobre `exp(cumsum(log r))`.

## 9. Tests (`tests/test_dashboard.py`)

Fixture sintética: 21 fondos, 3 años hábiles, 3 bloques correlacionados con categorías de `data/universe.json`
(mismos `fund_id`), seed fija. Fixture real (`data/cleaned/returns_matrix.parquet`) marcada `skipif` si falta.

1. **Smoke:** `create_app(returns)` arranca; para las 9 combinaciones `compute` devuelve `result` sin
   excepción y con `len(rows) == len(universo)`; `app.validation_layout` válido.
2. **Cache:** `monkeypatch` cuenta llamadas a `_compute_distance`. Secuencia: 3 perfiles × 3 métodos sobre la
   misma ventana → 1 llamada; cambiar lookback → 2; volver → sigue 2. Además `distance_bundle().linkage`
   `allclose` `optimize(..., "hrp", ...).linkage_matrix` y `leaf_order == result.leaf_order`.
3. **Callbacks (Flask test client, `POST /_dash-update-component`):** cambio de perfil altera `rowData`;
   cambio a HRP expone `raw_weight`; hover hoja → `selectedRows == {"ids": [fund]}`; hover enlace → ids ==
   subárbol (calculado independientemente desde `Z`); click → `pinned` contiene el fondo y hover-out mantiene
   `selectedRows`; segundo click → despinea.
4. **E2E (`dash_duo`, selenium):** hover real sobre un marcador de hoja → fila con clase seleccionada.
   `pytest.mark.skipif(shutil.which("chromedriver") is None)`.
5. Cobertura: `--cov=dashboard --cov-fail-under=85`.

## 10. Ejecución

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m dashboard.app            # http://127.0.0.1:8050
.venv/bin/python -m pytest tests/test_dashboard.py --cov=dashboard --cov-fail-under=85
```
