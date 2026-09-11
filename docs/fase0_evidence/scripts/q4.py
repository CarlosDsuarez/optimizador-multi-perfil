from soda import q
import pandas as pd, json
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_colwidth",60)
d = q("qhpu-8ixx","Q5 snapshot 2026-09-08",{"$where":"fecha_corte='2026-09-08T00:00:00.000'","$limit":5000}, quiet=True)
df = pd.DataFrame(d)
num = ["tipo_entidad","codigo_entidad","subtipo_negocio","codigo_negocio","principal_compartimento","tipo_participacion","valor_unidad_operaciones","valor_fondo_cierre_dia_t","numero_inversionistas","rentabilidad_diaria","rentabilidad_mensual","rentabilidad_semestral","rentabilidad_anual","numero_unidades_fondo_cierre","precierre_fondo_dia_t","rendimientos_abonados","aportes_recibidos","retiros_redenciones"]
for c in num: df[c]=pd.to_numeric(df[c], errors="coerce")
df.to_csv("snap_20260908.csv", index=False)
print("rows", len(df), "| distinct codigo_negocio", df.codigo_negocio.nunique(), "| distinct (codigo,participacion)", len(df[["codigo_negocio","tipo_participacion"]].drop_duplicates()), "| distinct (codigo,part,compart)", len(df[["codigo_negocio","tipo_participacion","principal_compartimento"]].drop_duplicates()))
print("principal_compartimento values:", df.principal_compartimento.value_counts().to_dict())
print("rows per codigo_negocio distribution:", df.groupby("codigo_negocio").size().value_counts().sort_index().to_dict())
print("\nsubtipo counts (rows / codes):")
print(df.groupby("nombre_subtipo_patrimonio").agg(rows=("codigo_negocio","size"), codes=("codigo_negocio","nunique"), aum_bn=("valor_fondo_cierre_dia_t", lambda s: round(s.sum()/1e9,1))))
print("\nentities:", df.nombre_entidad.nunique(), df.nombre_tipo_entidad.value_counts().to_dict())
# Example multi-class fund
multi = df.groupby("codigo_negocio").size(); ex = multi[multi>3].index[:2]
for code in ex:
    print(f"\n--- fund {code} share classes:")
    print(df[df.codigo_negocio==code][["nombre_entidad","nombre_patrimonio","principal_compartimento","tipo_participacion","valor_unidad_operaciones","valor_fondo_cierre_dia_t","numero_inversionistas","rentabilidad_diaria","rentabilidad_mensual","rentabilidad_anual"]].to_string())
# same code across different entities?
ce = df.groupby("codigo_negocio").codigo_entidad.nunique(); print("\ncodes present in >1 entity on same date:", int((ce>1).sum()))
# top 25 by AUM (fund level, principal_compartimento==1 if exists)
fund = df.sort_values("valor_fondo_cierre_dia_t", ascending=False).drop_duplicates("codigo_negocio")
print("\nTOP 30 by valor_fondo_cierre_dia_t (fund-level, largest row per code):")
print(fund.head(30)[["codigo_negocio","nombre_entidad","nombre_patrimonio","nombre_subtipo_patrimonio","valor_fondo_cierre_dia_t","numero_inversionistas","rentabilidad_anual"]].to_string())
