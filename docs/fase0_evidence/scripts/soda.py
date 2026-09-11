import requests, time, json, sys
BASE="https://www.datos.gov.co/resource/{}.json"
LOG=[]
def q(ds, label, params, show=40, quiet=False):
    t0=time.time()
    r=requests.get(BASE.format(ds), params=params, timeout=300)
    dt=time.time()-t0
    LOG.append((ds,label,params,r.status_code,dt))
    if r.status_code!=200:
        print(f"[{ds}] {label} STATUS {r.status_code} elapsed={dt:.2f}s\n  {r.text[:300]}"); return None
    data=r.json()
    if not quiet:
        print(f"[{ds}] {label}\n  GET {r.url}\n  rows={len(data)} elapsed={dt:.2f}s")
        for row in data[:show]: print("   ", json.dumps(row, ensure_ascii=False))
        print()
    return data
