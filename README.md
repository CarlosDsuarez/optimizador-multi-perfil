# Optimizador Multi-Perfil

Optimizador de portafolios sobre FICs colombianos (datos abiertos SFC / [datos.gov.co](https://www.datos.gov.co)). Soporta perfiles **conservador**, **moderado** y **agresivo** con tres metodologías: Markowitz, risk parity e HRP.

## Estructura

| Módulo | Descripción |
|---|---|
| [`data/`](data/README.md) | Ingesta y limpieza de series de valor de unidad |
| [`optimization/`](optimization/README.md) | Optimización con restricciones por perfil |
| [`backtest/`](backtest/README.md) | Walk-forward out-of-sample |
| [`dashboard/`](dashboard/README.md) | Dashboard interactivo (dendrograma HRP ↔ AG Grid) |

Reproducibilidad: cada corrida deja trazas en `manifest.json`.

## Instalación

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Opcional: `export SOCRATA_APP_TOKEN=...` para acelerar la descarga de datos.

## Uso rápido

```bash
# 1. Descargar y limpiar datos
.venv/bin/python -m data.ingest_sfc

# 2. Backtest walk-forward (base)
.venv/bin/python -m backtest.walk_forward

# 3. Dashboard interactivo
.venv/bin/python -m dashboard.app   # http://127.0.0.1:8050
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Datos

Los archivos crudos en `data/raw/` no se versionan (regenerables con `data.ingest_sfc`). El repo incluye `data/cleaned/` y resultados de backtest en `results/`.
