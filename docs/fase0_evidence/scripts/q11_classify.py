import pandas as pd, numpy as np, re
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60); pd.set_option("display.max_rows",400)
df=pd.read_csv("monthly_all.csv"); df["fecha_corte"]=pd.to_datetime(df.fecha_corte)
for c in ["valor_unidad_operaciones","valor_fondo_cierre_dia_t","tipo_participacion","codigo_negocio","codigo_entidad"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.drop_duplicates(["fecha_corte","codigo_negocio","tipo_participacion"])
df["nombre_norm"]=df.nombre_patrimonio.str.upper().str.replace(r"\s+"," ",regex=True).str.strip()
# fund identity = normalized name (stitches code handoffs); per (name, class) monthly VU series
df["key"]=df.nombre_norm+"||"+df.tipo_participacion.astype(int).astype(str)
piv=df.pivot_table(index="fecha_corte",columns="key",values="valor_unidad_operaciones",aggfunc="first")
piv=piv.replace(0,np.nan)
rets=np.log(piv).diff()
colcap=rets["FONDO BURSÁTIL ISHARES MSCI COLCAP||800"]; glob=rets["CREDICORP CAPITAL ACCIONES GLOBALES||800"]; mm=rets["FONDO DE INVERSIÓN COLECTIVA ABIERTO FIDUCUENTA||800"]
out=[]
for k in piv.columns:
    s=piv[k].dropna()
    if len(s)<24: continue
    r=rets[k].dropna()
    name,cls=k.split("||")
    last=s.index[-1]; first=s.index[0]
    sub=df[df.key==k]
    out.append(dict(nombre=name,clase=int(cls),first=first.date(),last=last.date(),months=len(s),alive=last>=pd.Timestamp("2026-09-01"),
        vol_m=float(r.std()*np.sqrt(12)*100),ret_a=float((s.iloc[-1]/s.iloc[0])**(12/len(s))-1)*100,
        corr_colcap=float(r.corr(colcap)),corr_glob=float(r.corr(glob)),corr_mm=float(r.corr(mm)),
        beta_colcap=float(r.cov(colcap)/colcap.var()) if colcap.var()>0 else np.nan,
        maxdd=float(((s/s.cummax())-1).min()*100),
        aum_bn=float(sub.valor_fondo_cierre_dia_t.iloc[-1]/1e9),codes="+".join(map(str,sorted(sub.codigo_negocio.unique()))),subtipo=sub.nombre_subtipo_patrimonio.iloc[-1],entidad=sub.nombre_entidad.iloc[-1][:28]))
m=pd.DataFrame(out)
# fund-level: keep class with most months then largest AUM
m=m.sort_values(["nombre","months","aum_bn"],ascending=[True,False,False]).drop_duplicates("nombre")
# fund total AUM at last date
aum_fund=df[df.fecha_corte==df.fecha_corte.max()].groupby("nombre_norm").valor_fondo_cierre_dia_t.sum()/1e9
m["aum_fund_bn"]=m.nombre.map(aum_fund).fillna(0)
def vclass(r):
    if r.vol_m<2.0: return "RF_CORTO/MM"
    if r.vol_m<8.0 and r.corr_colcap<0.5 and r.corr_glob<0.5: return "RF_MEDIO_LARGO"
    if r.corr_colcap>=0.8: return "RV_LOCAL"
    if r.corr_glob>=0.5 and r.corr_colcap<0.6: return "RV_INTL/GLOBAL"
    if r.vol_m>=8: return "RV/ALTERNATIVO_OTRO"
    return "MIXTO"
m["clase_vol"]=m.apply(vclass,axis=1)
m=m.sort_values(["clase_vol","aum_fund_bn"],ascending=[True,False]); m.to_csv("classified_all.csv",index=False)
print("funds (by name) with >=24 months:", len(m), "| alive:", int(m.alive.sum()))
print(m.groupby(["clase_vol","alive"]).size().unstack(fill_value=0))
alive=m[m.alive & (m.first<=pd.Timestamp("2018-06-30").date())]
print("\nALIVE & history since <=2018-06 by vol-class:"); print(alive.clase_vol.value_counts())
cols=["nombre","clase","codes","first","months","vol_m","ret_a","maxdd","corr_colcap","corr_glob","corr_mm","aum_fund_bn","subtipo","entidad"]
for c in ["RF_MEDIO_LARGO","RV_LOCAL","RV_INTL/GLOBAL","MIXTO","RV/ALTERNATIVO_OTRO"]:
    sub=alive[alive.clase_vol==c].head(30)
    print(f"\n=== {c}: alive, first<=2018-06 (top {len(sub)} by fund AUM) ===")
    print(sub[cols].round(2).to_string(index=False))
sub=alive[alive.clase_vol=="RF_CORTO/MM"].head(12)
print(f"\n=== RF_CORTO/MM: alive, first<=2018-06 (top 12 by AUM) ===");print(sub[cols].round(2).to_string(index=False))
# Also younger but relevant (first 2018-07..2021-12) for RF_MEDIO_LARGO, MIXTO, RV_INTL
young=m[m.alive & (m.first>pd.Timestamp("2018-06-30").date()) & (m.first<=pd.Timestamp("2021-12-31").date())]
for c in ["RF_MEDIO_LARGO","MIXTO","RV_INTL/GLOBAL","RV_LOCAL"]:
    sub=young[young.clase_vol==c].head(12)
    print(f"\n=== YOUNGER (first 2018-07..2021-12) {c} ===");print(sub[cols].round(2).to_string(index=False))
