"""Build candidate list from latest snapshot: FIC only (exclude FCP), aggregate AUM per fund, name-based category heuristic."""
import pandas as pd, numpy as np, re
pd.set_option("display.width",260); pd.set_option("display.max_columns",40); pd.set_option("display.max_colwidth",75); pd.set_option("display.max_rows",300)
snap = pd.read_csv("snap_20260908.csv").drop_duplicates(["codigo_negocio","tipo_participacion"])
fic = snap[snap.subtipo_negocio!=7].copy()
agg = fic.groupby("codigo_negocio").agg(entidad=("nombre_entidad","first"), nombre=("nombre_patrimonio","first"), subtipo=("nombre_subtipo_patrimonio","first"),
      aum=("valor_fondo_cierre_dia_t","sum"), inversionistas=("numero_inversionistas","sum"), n_clases=("tipo_participacion","nunique"), rent_anual_med=("rentabilidad_anual","median")).reset_index()
main = fic.sort_values("valor_fondo_cierre_dia_t",ascending=False).drop_duplicates("codigo_negocio")[["codigo_negocio","tipo_participacion","valor_unidad_operaciones"]].rename(columns={"tipo_participacion":"clase_principal"})
agg = agg.merge(main, on="codigo_negocio")
def cat(name):
    n=name.upper()
    if re.search(r"ACCIONES|ACCION|COLCAP|BURS[AÁ]TIL|EQUITY|S&P|MSCI|ISHARES|GLOBAL X", n):
        return "RV_INTL" if re.search(r"GLOBAL|INTERNACIONAL|LATAM|USA|EEUU|MUNDIAL|EMERGENTES|ASIA|EUROPA|S&P ?500|NASDAQ|MSCI WORLD", n) and not re.search(r"COLCAP|COLOMBIA SELECT", n) else "RV_LOCAL"
    if re.search(r"GLOBAL|INTERNACIONAL|D[OÓ]LAR|USD|OFFSHORE|MUNDIAL|EMERGENTES", n): return "INTL_OTRO"
    if re.search(r"BALANCEAD|MIXT|MODERAD|CRECIMIENTO|DIVERSIFICAD|MULTIACTIVO|ESTRATEGI|PERFIL|CONSERVADOR|AGRESIVO", n): return "MIXTO"
    if re.search(r"LIQUIDEZ|EFECTIVO|VISTA|A LA VISTA|CASH|TESORER|MONETARIO|RENTA L[IÍ]QUIDA|INTER[EÉ]S|RENTAF[AÁ]CIL|FIDUCUENTA|1525|OCCITESOROS|SUMAR|ABIERT[OA] SIN PACTO|DIGITAL|EXCEDENTES|PA[IÍ]S|CORTO PLAZO", n): return "RF_CORTO"
    if re.search(r"PLAZO|PERMANENCIA|RENTA FIJA|LARGO|DEUDA|TES|CR[EÉ]DITO|BONOS|RENTA ALTA|MEDIANO|RENTABILIDAD|GOB|RENTA VALOR", n): return "RF_LARGO"
    return "SIN_CLASIFICAR"
agg["cat_nombre"]=agg.nombre.apply(cat)
agg = agg.sort_values("aum",ascending=False)
agg["aum_bn_cop"]=(agg.aum/1e9).round(1)
agg.to_csv("candidates_snapshot.csv",index=False)
print("FIC funds (ex-FCP) on 2026-09-08:", len(agg), "| total AUM (bn COP):", round(agg.aum.sum()/1e9,0))
print(agg.cat_nombre.value_counts())
print("\nAUM share by name-category:"); print((agg.groupby("cat_nombre").aum.sum()/agg.aum.sum()*100).round(1))
for c in ["RV_LOCAL","RV_INTL","INTL_OTRO","MIXTO","RF_LARGO","RF_CORTO","SIN_CLASIFICAR"]:
    sub=agg[agg.cat_nombre==c].head(25)
    print(f"\n=== {c} (top {len(sub)} by AUM) ===")
    print(sub[["codigo_negocio","clase_principal","entidad","nombre","subtipo","aum_bn_cop","inversionistas","n_clases","rent_anual_med"]].to_string(index=False))
