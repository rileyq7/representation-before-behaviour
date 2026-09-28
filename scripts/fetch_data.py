"""Re-fetch public data and require byte identity with the original manifest."""
import hashlib
import json
from pathlib import Path
import urllib.request
root=Path(__file__).resolve().parents[1]
for item in json.loads((root / "sources/data-manifest.json").read_text()):
    path=root/item["file"]
    data=path.read_bytes() if path.exists() else urllib.request.urlopen(item["url"]).read()
    if hashlib.sha256(data).hexdigest()!=item["sha256"]:
        raise RuntimeError(f"Source changed: {item['url']}. Do not silently mix versions.")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    print(f"Verified {item['file']}")
