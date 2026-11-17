import json as _j
JSONDecodeError = _j.JSONDecodeError
loads = lambda s, *a, **k: _j.loads(s)
load = lambda fp, *a, **k: _j.load(fp)
dumps = lambda o, *a, **k: _j.dumps(o, **{x: y for x, y in k.items() if x in ("ensure_ascii", "indent", "sort_keys", "default")})
dump = lambda o, fp, *a, **k: _j.dump(o, fp, **{x: y for x, y in k.items() if x in ("ensure_ascii", "indent", "sort_keys", "default")})
