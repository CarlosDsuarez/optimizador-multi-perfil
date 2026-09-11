import json, sys, time
from sodapy import Socrata
c = Socrata("www.datos.gov.co", None, timeout=60)
for ds in ["qhpu-8ixx", "djw7-ur7t"]:
    t0 = time.time()
    m = c.get_metadata(ds)
    print("="*80)
    print("DATASET", ds, "|", m.get("name"), "| rows:", m.get("rowsUpdatedAt"), "| created:", m.get("createdAt"), "| updated:", m.get("rowsUpdatedAt"))
    print("description:", (m.get("description") or "")[:1500])
    print("attribution:", m.get("attribution"), "| category:", m.get("category"))
    print("metadata.custom_fields:", json.dumps(m.get("metadata", {}).get("custom_fields", {}), ensure_ascii=False)[:800])
    print("COLUMNS:")
    for col in m["columns"]:
        cs = col.get("cachedContents", {})
        print(f"  - {col['fieldName']:<35} {col['dataTypeName']:<12} name='{col['name']}' | desc='{(col.get('description') or '')[:120]}' | non_null={cs.get('non_null')} null={cs.get('null')} cardinality={cs.get('cardinality')} min={cs.get('smallest')} max={cs.get('largest')}")
    print("elapsed", round(time.time()-t0,2))
