from soda import q
import pandas as pd, time
frames=[]
for y in range(2016,2027):
    d = q("qhpu-8ixx",f"Q7b lifetimes {y}",{"$select":"codigo_negocio, nombre_patrimonio, codigo_entidad, nombre_subtipo_patrimonio, min(fecha_corte) as fmin, max(fecha_corte) as fmax, count(*) as n",
        "$where":f"fecha_corte between '{y}-01-01T00:00:00' and '{y}-12-31T23:59:59'","$group":"codigo_negocio, nombre_patrimonio, codigo_entidad, nombre_subtipo_patrimonio","$limit":50000}, quiet=True)
    if d is None: print("FAILED", y); continue
    df=pd.DataFrame(d); df["anio"]=y; frames.append(df); print(y, len(df), "combos")
all_=pd.concat(frames); all_.to_csv("lifetimes_by_year.csv",index=False); print("saved", len(all_))
