import pandas as pd, numpy as np, re
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80)
df=pd.read_csv("lifetimes_by_year.csv"); df["fmin"]=pd.to_datetime(df.fmin); df["fmax"]=pd.to_datetime(df.fmax)
df["nombre_norm"]=df.nombre_patrimonio.str.upper().str.replace(r"\s+"," ",regex=True).str.strip()
g=df.groupby(["codigo_negocio","nombre_norm","codigo_entidad","nombre_subtipo_patrimonio"]).agg(fmin=("fmin","min"),fmax=("fmax","max"),n=("n","sum")).reset_index()
print("distinct codes overall:", g.codigo_negocio.nunique(), "| distinct names:", g.nombre_norm.nunique(), "| combos:", len(g))
cn=g.groupby("codigo_negocio").nombre_norm.nunique(); print("codes with >1 name (rename under same code):", int((cn>1).sum()))
ce=g.groupby("codigo_negocio").codigo_entidad.nunique(); print("codes with >1 entity (manager change under same code):", int((ce>1).sum()))
cs=g.groupby("codigo_negocio").nombre_subtipo_patrimonio.nunique(); print("codes with >1 subtipo:", int((cs>1).sum()))
nc=g.groupby("nombre_norm").codigo_negocio.nunique(); print("names with >1 code:", int((nc>1).sum()))
# Sequential reassignment: same name, code A ends and code B starts within 90 days
life=g.groupby(["codigo_negocio"]).agg(fmin=("fmin","min"),fmax=("fmax","max"),n=("n","sum")).reset_index()
life["days"]=(life.fmax-life.fmin).dt.days+1
print("\nSEQUENTIAL SAME-NAME CODE HANDOFFS (code A ends, code B with same name starts within ±90d):")
cnt=0
for name in nc[nc>1].index:
    sub=g[g.nombre_norm==name].groupby("codigo_negocio").agg(fmin=("fmin","min"),fmax=("fmax","max"),ent=("codigo_entidad","first")).sort_values("fmin").reset_index()
    for i in range(len(sub)-1):
        gap=(sub.loc[i+1,"fmin"]-sub.loc[i,"fmax"]).days
        if -90<=gap<=90:
            cnt+=1
            if cnt<=25: print(f"  '{name[:60]}': {sub.loc[i,'codigo_negocio']} (ent {sub.loc[i,'ent']}) ends {sub.loc[i,'fmax'].date()} -> {sub.loc[i+1,'codigo_negocio']} (ent {sub.loc[i+1,'ent']}) starts {sub.loc[i+1,'fmin'].date()} gap={gap}d")
print("  total handoffs detected:", cnt)
print("\nCODES WITH >1 NAME examples (first 15):")
for code in cn[cn>1].index[:15]:
    sub=g[g.codigo_negocio==code].sort_values("fmin")
    print(f"  code {code}: " + " || ".join(f"'{r.nombre_norm[:55]}' {r.fmin.date()}..{r.fmax.date()}" for _,r in sub.iterrows()))
print("\nCODES WITH >1 ENTITY examples:")
for code in ce[ce>1].index[:10]:
    sub=g[g.codigo_negocio==code].sort_values("fmin")
    print(f"  code {code}: " + " || ".join(f"ent={r.codigo_entidad} '{r.nombre_norm[:45]}' {r.fmin.date()}..{r.fmax.date()}" for _,r in sub.iterrows()))
# Survivorship
alive=life[life.fmax>=pd.Timestamp("2026-09-01")]
print("\nCodes alive 2026-09:", len(alive), "| of which start<=2016-01-05:", int((alive.fmin<=pd.Timestamp("2016-01-05")).sum()))
for y in range(2016,2025): print(f"  alive & history since <= {y}-01-31: {int((alive.fmin<=pd.Timestamp(f'{y}-01-31')).sum())}")
dead=life[life.fmax<pd.Timestamp("2026-09-01")]; print("Codes dead (last obs before 2026-09):", len(dead), "| died per year:", dead.groupby(dead.fmax.dt.year).size().to_dict())
# Codes with internal gaps: sum of yearly n vs calendar days in [fmin,fmax] — proxy for discontinuity (over-count due to classes so use distinct dates later)
life.to_csv("code_lifetimes.csv",index=False)
# Code numbering vs start date
life["bucket"]=pd.cut(life.codigo_negocio,[0,10000,20000,60000,100000,130000,200000])
print("\ncode bucket vs first-observed date:"); print(life.groupby("bucket",observed=True).agg(n=("codigo_negocio","size"),fmin_min=("fmin","min"),fmin_med=("fmin","median"),fmin_max=("fmin","max")))
