# `data/` — capa de ingesta y limpieza (Fase 1)

| Ruta | Qué es |
|---|---|
| `universe.json` | Universo aprobado en Fase 0 (D3): 18 `core` + 3 `late`, con `codigo_negocio` encadenados, clase (`tipo_participacion`) y umbrales de suficiencia. Su SHA256 va al manifest. |
| `ingest_sfc.py` | Pipeline: `SodaClient` (retries/backoff explícitos) → `fetch_raw` → `build_clean` → `write_outputs` + `manifest.json`. |
| `raw/qhpu-8ixx_<code>.parquet` | Respuesta cruda de la API por `codigo_negocio` (todas las clases, 26 columnas, valores como strings). **No versionado**; regenerable. |
| `raw/djw7-ur7t_metadata.parquet` | Última fila por fondo del dataset espejo (cross-check de VU + nombre/entidad/subtipo). |
| `cleaned/returns_matrix.parquet` | Log-retornos diarios sobre calendario hábil colombiano (`holidays.CO`), columnas = `fund_id`. |
| `cleaned/prices_matrix.parquet` | `valor_unidad_operaciones` alineado al mismo calendario (para verificar / reconstruir). |
| `cleaned/funds_master.parquet` | Tabla maestra (D1): `fund_id`, categoría, rol, cadena de códigos, clase, primera/última fecha, metadata de la API. |

## Uso

```bash
export SOCRATA_APP_TOKEN=...        # opcional (D11); sin token funciona pero puede throttlear
python -m data.ingest_sfc            # descarga + limpia + manifest.json
python -m data.ingest_sfc --skip-download   # re-limpia a partir de data/raw
python -m pytest tests/test_ingest_sfc.py --cov=data.ingest_sfc
```

Exit codes: `0` OK · `2` `InsufficientDataError` (un fondo del universo contradice Fase 0 → **se detiene, no sustituye**) · `1` otro `IngestError` (HTTP agotado, datos inconsistentes).

## Política de NaN (también en `manifest.json → returns.nan_policy`)

- `ret_t = log(VU_t / VU_prev)` con `prev` = último día hábil **observado**.
- Antes del primer dato / después del último → NaN (fuera del universo; point-in-time downstream).
- Día hábil sin dato dentro de la vida del fondo → NaN, **sin imputación**.
- Primer día tras una brecha → retorno acumulado de la brecha (NAV continuo); cada celda queda listada en `manifest.json → gap_log` con `missing_bdays` para que downstream pueda enmascararla.
- Fin de semana / festivo → se pliega al siguiente día hábil (devengo de fondos monetarios cae en lunes).

## Encadenamiento de códigos (D1)

`splice_codes` exige: sin solape, brecha ≤ 1 día, `|VU_B/VU_A − 1| ≤ 2 %`. Cada empalme aplicado queda en `manifest.json → code_mappings` con fechas y VUs observados. Único caso en el universo: `43502 → 129433` (Global X Colombia Select, 2026-01-01).
