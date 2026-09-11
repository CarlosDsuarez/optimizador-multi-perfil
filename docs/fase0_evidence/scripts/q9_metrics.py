import pandas as pd, numpy as np, json
pd.set_option("display.width",260); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
chains=json.load(open("chains.json"))
END=pd.Timestamp("2026-09-08")
def load(code):
    df=pd.read_csv(f"hist/{code}.csv"); df["fecha_corte"]=pd.to_datetime(df.fecha_corte)
    for c in ["valor_unidad_operaciones","valor_fondo_cierre_dia_t","numero_inversionistas","rentabilidad_diaria","tipo_participacion"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df
rows=[]; series={}
for head,info in chains.items():
    parts=[load(c) for c in info["codes"]]; df=pd.concat(parts)
    n_dup=int(df.duplicated(["fecha_corte","codigo_negocio","tipo_participacion"]).sum()); n_raw=len(df)
    df=df.drop_duplicates(["fecha_corte","codigo_negocio","tipo_participacion"])
    # pick class with longest history (distinct dates) across chain; tie -> larger last AUM
    cl=df.groupby("tipo_participacion").agg(nd=("fecha_corte","nunique"),aum=("valor_fondo_cierre_dia_t","last")).sort_values(["nd","aum"],ascending=False)
    cls=int(cl.index[0]); s=df[df.tipo_participacion==cls].sort_values("fecha_corte")
    # handoff check
    handoff=""
    if len(info["codes"])>1:
        for a,b in zip(info["codes"][:-1],info["codes"][1:]):
            sa=s[s.codigo_negocio==a]; sb=s[s.codigo_negocio==b]
            if len(sa) and len(sb):
                handoff+=f"{a}->{b}: VU {sa.valor_unidad_operaciones.iloc[-1]:.2f}@{sa.fecha_corte.iloc[-1].date()} -> {sb.valor_unidad_operaciones.iloc[0]:.2f}@{sb.fecha_corte.iloc[0].date()}; "
            else: handoff+=f"{a}->{b}: class {cls} missing in one side; "
    s=s.drop_duplicates("fecha_corte").set_index("fecha_corte")
    vu=s.valor_unidad_operaciones.replace(0,np.nan)
    first,last=vu.first_valid_index(),vu.last_valid_index()
    cal=pd.date_range(first,last); missing=cal.difference(vu.dropna().index)
    gaps=pd.Series(vu.dropna().index).diff().dt.days; maxgap=int(gaps.max()) if len(gaps) else 0; ngaps=int((gaps>1).sum())
    # business-day series
    bd=vu.reindex(pd.bdate_range(first,last)).ffill()
    r=np.log(bd).diff().dropna()
    zero_bd=float((r.abs()<1e-9).mean()*100)
    vol=float(r.std()*np.sqrt(252)*100); ann=float((bd.iloc[-1]/bd.iloc[0])**(252/len(bd))-1)*100
    dd=float(((bd/bd.cummax())-1).min()*100)
    # bad values
    bad=int((s.valor_unidad_operaciones<=0).sum()); nan_vu=int(s.valor_unidad_operaciones.isna().sum())
    big=int((r.abs()>0.15).sum())
    aum_last=float(df[df.fecha_corte==df.fecha_corte.max()].valor_fondo_cierre_dia_t.sum()/1e9)
    series[head]=r
    rows.append(dict(head=head,cat=info["cat"],codes="+".join(map(str,info["codes"])),clase=cls,n_clases=int(df.tipo_participacion.nunique()),nombre=info["names"][0][:55],
        first=first.date(),last=last.date(),years=round((last-first).days/365.25,1),n_obs=int(vu.notna().sum()),cal_days=len(cal),missing_pct=round(len(missing)/len(cal)*100,2),
        n_gaps=ngaps,max_gap_d=maxgap,dups=n_dup,vu_le0=bad,vu_nan=nan_vu,zero_ret_bd_pct=round(zero_bd,1),jumps_gt15pct=big,vol_ann=round(vol,1),ret_ann=round(ann,1),maxdd=round(dd,1),aum_bn=round(aum_last,1),alive=(last>=END-pd.Timedelta(days=3)),handoff=handoff))
m=pd.DataFrame(rows).sort_values(["cat","first"]); m.to_csv("metrics.csv",index=False)
print(m.drop(columns=["handoff","codes"]).to_string(index=False))
print("\nHANDOFF continuity checks:"); [print(" ",r.head,r.handoff) for r in m.itertuples() if r.handoff]
# correlations vs COLCAP ETF & Fiducuenta, weekly returns to reduce async noise
R=pd.DataFrame(series); W=R.resample("W-FRI").sum(); W=W.loc["2019":]
c=W.corr()
print("\nCorr (weekly, 2019+) vs iShares COLCAP (60678) and vs Credicorp Acciones Globales (58918):")
print(pd.DataFrame({"cat":m.set_index("head").cat,"corr_COLCAP":c["60678"].round(2),"corr_ACC_GLOB":c["58918"].round(2),"corr_FIDUCUENTA":c["2852"].round(2),"corr_FIDUGOB":c["128979"].round(2)}).sort_values(["cat","corr_COLCAP"]).to_string())
R.to_pickle("returns_bd.pkl")
