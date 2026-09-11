import json, time, logging
logging.disable(logging.WARNING)
from sodapy import Socrata
c = Socrata("www.datos.gov.co", None, timeout=120)

def q(ds, label, **kw):
    t0=time.time()
    try:
        r = c.get(ds, **kw)
    except Exception as e:
        print(f"[{ds}] {label}\n  params={kw}\n  ERROR: {e}\n"); return None
    dt=time.time()-t0
    print(f"[{ds}] {label}\n  params={json.dumps(kw, ensure_ascii=False)}\n  rows={len(r)} elapsed={dt:.2f}s")
    for row in r[:40]: print("   ", json.dumps(row, ensure_ascii=False))
    print()
    return r

for ds in ["qhpu-8ixx","djw7-ur7t"]:
    q(ds, "Q1 rango fechas + conteos", select="min(fecha_corte) as fmin, max(fecha_corte) as fmax, count(*) as n, count(distinct codigo_negocio) as n_codigos, count(distinct fecha_corte) as n_fechas, count(distinct nombre_patrimonio) as n_nombres, count(distinct codigo_entidad) as n_entidades")
    q(ds, "Q2 subtipos", select="subtipo_negocio, nombre_subtipo_patrimonio, count(*) as n, count(distinct codigo_negocio) as n_codigos", group="subtipo_negocio, nombre_subtipo_patrimonio", order="subtipo_negocio")
    q(ds, "Q3 fechas distintas por año", select="date_trunc_y(fecha_corte) as anio, count(distinct fecha_corte) as n_fechas, count(*) as n_rows, count(distinct codigo_negocio) as n_codigos", group="anio", order="anio")
