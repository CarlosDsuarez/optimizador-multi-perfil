"""Pull full history (all classes) for candidate funds, stitching predecessor codes by normalized name."""
import pandas as pd, numpy as np, json, time, os
from soda import q
CANDS = {
 # RF_CORTO
 2852:"RF_CORTO", 58756:"RF_CORTO", 11407:"RF_CORTO", 59304:"RF_CORTO", 8734:"RF_CORTO", 129120:"RF_CORTO", 58347:"RF_CORTO", 10779:"RF_CORTO", 128956:"RF_CORTO", 18462:"RF_CORTO", 119133:"RF_CORTO",
 # RF_LARGO
 128979:"RF_LARGO", 10824:"RF_LARGO", 2971:"RF_LARGO", 3049:"RF_LARGO", 58883:"RF_LARGO", 3814:"RF_LARGO", 59076:"RF_LARGO", 58897:"RF_LARGO", 63301:"RF_LARGO", 131893:"RF_LARGO", 78886:"RF_LARGO", 59741:"RF_LARGO",
 # RV_LOCAL
 60678:"RV_LOCAL", 129433:"RV_LOCAL", 59105:"RV_LOCAL", 67403:"RV_LOCAL", 58735:"RV_LOCAL", 59738:"RV_LOCAL", 119158:"RV_LOCAL", 11014:"RV_LOCAL",
 # RV_INTL
 58918:"RV_INTL", 3644:"RV_INTL", 99106:"RV_INTL", 59079:"RV_INTL", 59740:"RV_INTL", 123532:"RV_INTL", 95589:"RV_INTL", 69368:"RV_INTL",
 # MIXTO
 53962:"MIXTO", 3078:"MIXTO", 115127:"MIXTO", 105784:"MIXTO", 59078:"MIXTO", 93223:"MIXTO", 113658:"MIXTO", 101366:"MIXTO",
}
lt=pd.read_csv("lifetimes_by_year.csv"); lt["nombre_norm"]=lt.nombre_patrimonio.str.upper().str.replace(r"\s+"," ",regex=True).str.strip()
chains={}
for code,cat in CANDS.items():
    names=set(lt[lt.codigo_negocio==code].nombre_norm)
    codes=sorted(set(lt[lt.nombre_norm.isin(names)].codigo_negocio))
    chains[code]={"cat":cat,"names":sorted(names),"codes":codes}
    if len(codes)>1: print(f"  chain for {code}: {codes}  names={[n[:50] for n in names]}")
json.dump(chains, open("chains.json","w"), ensure_ascii=False, indent=1)
os.makedirs("hist",exist_ok=True)
allcodes=sorted({c for v in chains.values() for c in v["codes"]})
print("total codes to pull:", len(allcodes))
for code in allcodes:
    fn=f"hist/{code}.csv"
    if os.path.exists(fn): continue
    rows=[]; off=0
    while True:
        d=q("qhpu-8ixx",f"hist {code}",{"$select":"fecha_corte,codigo_negocio,codigo_entidad,nombre_entidad,nombre_patrimonio,nombre_subtipo_patrimonio,principal_compartimento,tipo_participacion,valor_unidad_operaciones,valor_fondo_cierre_dia_t,numero_inversionistas,rentabilidad_diaria,rentabilidad_anual",
            "$where":f"codigo_negocio={code}","$order":"fecha_corte,tipo_participacion","$limit":50000,"$offset":off}, quiet=True)
        if d is None: print("FAILED", code); break
        rows+=d
        if len(d)<50000: break
        off+=50000
    pd.DataFrame(rows).to_csv(fn,index=False); print(f"  {code}: {len(rows)} rows")
print("done")
