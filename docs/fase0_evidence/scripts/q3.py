from soda import q
import pandas as pd, json
# Distinct dates with row counts (all years)
d = q("qhpu-8ixx","Q4 fechas distintas",{"$select":"fecha_corte, count(*) as n","$group":"fecha_corte","$order":"fecha_corte","$limit":50000}, quiet=True)
df = pd.DataFrame(d); df["fecha_corte"]=pd.to_datetime(df["fecha_corte"]); df["n"]=df["n"].astype(int)
df["dow"]=df["fecha_corte"].dt.dayofweek
print("distinct dates:", len(df), "| range", df.fecha_corte.min().date(), "->", df.fecha_corte.max().date())
print("calendar days in range:", (df.fecha_corte.max()-df.fecha_corte.min()).days+1)
print("by year: distinct dates / calendar days / weekend dates / median rows per date")
for y,g in df.groupby(df.fecha_corte.dt.year):
    cal = pd.date_range(f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), df.fecha_corte.max()))
    print(f"  {y}: {len(g):4d} / {len(cal):3d} / weekend={int((g.dow>=5).sum()):3d} / med_rows={int(g.n.median()):5d} min_rows={g.n.min():5d} max_rows={g.n.max():5d}")
# rows per weekday overall
print("median rows per date by weekday (0=Mon):", df.groupby("dow").n.median().to_dict())
# gaps
full = pd.date_range(df.fecha_corte.min(), df.fecha_corte.max())
missing = full.difference(df.fecha_corte)
print("missing calendar dates:", len(missing), list(missing[:30].strftime("%Y-%m-%d")))
df.to_csv("dates.csv", index=False)
