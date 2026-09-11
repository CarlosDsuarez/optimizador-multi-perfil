from soda import q
import pandas as pd, numpy as np
rows=[]; off=0
while True:
    d=q("qhpu-8ixx","Q10 month-start VU all FIC",{"$select":"fecha_corte,codigo_negocio,codigo_entidad,nombre_entidad,nombre_patrimonio,nombre_subtipo_patrimonio,tipo_participacion,valor_unidad_operaciones,valor_fondo_cierre_dia_t,numero_inversionistas",
        "$where":"date_extract_d(fecha_corte)=1 AND subtipo_negocio!=7","$order":"fecha_corte,codigo_negocio,tipo_participacion","$limit":50000,"$offset":off}, quiet=True)
    rows+=d; print("page", off, len(d))
    if len(d)<50000: break
    off+=50000
df=pd.DataFrame(rows); df.to_csv("monthly_all.csv",index=False); print("rows", len(df), "codes", df.codigo_negocio.nunique())
