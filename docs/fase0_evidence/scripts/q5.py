from soda import q
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
snap = pd.read_csv("snap_20260908.csv")
dup = snap[snap.duplicated(["codigo_negocio","tipo_participacion"], keep=False)].sort_values(["codigo_negocio","tipo_participacion"])
print("DUPLICATE KEYS on 2026-09-08:"); print(dup[["codigo_entidad","nombre_entidad","codigo_negocio","nombre_patrimonio","principal_compartimento","tipo_participacion","valor_unidad_operaciones","valor_fondo_cierre_dia_t","rentabilidad_diaria"]].to_string())

def series(code, part, start, end):
    d = q("qhpu-8ixx",f"Q6 serie {code}/{part}",{"$where":f"codigo_negocio={code} AND tipo_participacion={part} AND fecha_corte between '{start}' and '{end}'","$order":"fecha_corte","$limit":5000,"$select":"fecha_corte,codigo_negocio,tipo_participacion,valor_unidad_operaciones,valor_fondo_cierre_dia_t,rentabilidad_diaria,rentabilidad_mensual,rentabilidad_semestral,rentabilidad_anual,numero_inversionistas"}, quiet=True)
    df = pd.DataFrame(d); 
    for c in df.columns[1:]: df[c]=pd.to_numeric(df[c])
    df["fecha_corte"]=pd.to_datetime(df.fecha_corte); df["dow"]=df.fecha_corte.dt.day_name().str[:3]
    df["r_simple_pct"] = df.valor_unidad_operaciones.pct_change()*100
    df["r_ea365_pct"] = ((1+df.valor_unidad_operaciones.pct_change())**365-1)*100
    df["r_ea360_pct"] = ((1+df.valor_unidad_operaciones.pct_change())**360-1)*100
    return df

# Fiducuenta main class
snap_f = snap[snap.codigo_negocio==2852].sort_values("valor_fondo_cierre_dia_t",ascending=False)
print("\nFiducuenta classes:\n", snap_f[["tipo_participacion","valor_unidad_operaciones","valor_fondo_cierre_dia_t","numero_inversionistas","rentabilidad_diaria","rentabilidad_anual"]].to_string())
part = int(snap_f.iloc[0].tipo_participacion)
df = series(2852, part, "2026-08-20", "2026-09-08")
print(f"\nFIDUCUENTA 2852/{part}:\n", df[["fecha_corte","dow","valor_unidad_operaciones","rentabilidad_diaria","r_simple_pct","r_ea365_pct","r_ea360_pct","rentabilidad_mensual","rentabilidad_anual"]].round(6).to_string())

# iShares COLCAP ETF
snap_e = snap[snap.codigo_negocio==60678]
part = int(snap_e.iloc[0].tipo_participacion)
df = series(60678, part, "2026-08-20", "2026-09-08")
print(f"\nISHARES COLCAP 60678/{part}:\n", df[["fecha_corte","dow","valor_unidad_operaciones","rentabilidad_diaria","r_simple_pct","r_ea365_pct","rentabilidad_mensual","rentabilidad_anual"]].round(4).to_string())
# verify rentabilidad_anual = (VU_t/VU_{t-365})-1 ?
df2 = series(60678, part, "2025-08-01", "2026-09-08")
df2 = df2.set_index("fecha_corte")
t = pd.Timestamp("2026-09-08"); 
for lag in [365,366,364,360,252]:
    t0 = t - pd.Timedelta(days=lag)
    if t0 in df2.index:
        print(f"  anual check lag {lag}d: VU ratio-1 = {(df2.loc[t,'valor_unidad_operaciones']/df2.loc[t0,'valor_unidad_operaciones']-1)*100:.4f}% vs reported {df2.loc[t,'rentabilidad_anual']:.4f}")
for lag in [30,31,28]:
    t0 = t - pd.Timedelta(days=lag)
    if t0 in df2.index:
        r=(df2.loc[t,'valor_unidad_operaciones']/df2.loc[t0,'valor_unidad_operaciones'])
        print(f"  mensual check lag {lag}d: simple {(r-1)*100:.4f}% | EA {((r)**(365/lag)-1)*100:.4f}% vs reported {df2.loc[t,'rentabilidad_mensual']:.4f}")
for lag in [180,182,183]:
    t0 = t - pd.Timedelta(days=lag)
    if t0 in df2.index:
        r=(df2.loc[t,'valor_unidad_operaciones']/df2.loc[t0,'valor_unidad_operaciones'])
        print(f"  semestral check lag {lag}d: simple {(r-1)*100:.4f}% | EA {((r)**(365/lag)-1)*100:.4f}% vs reported {df2.loc[t,'rentabilidad_semestral']:.4f}")
