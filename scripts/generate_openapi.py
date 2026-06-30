import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from app.main import app

spec = app.openapi()
with open("openapi_spec.json", "w") as f:
    json.dump(spec, f, indent=2)

print(f"OpenAPI spec saved: {len(json.dumps(spec))} bytes, {len(spec['paths'])} paths")
for path, methods in spec["paths"].items():
    for method in methods:
        print(f"  {method.upper()} {path}")
