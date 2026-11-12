"""
Writes the phone app's data files (shared by every sport's export_app.py).

When the APP_PASSWORD environment variable is set (it is on the hosted build), every file is
encrypted with AES-256-GCM using a key derived from the password (PBKDF2-SHA256), and
lock.json lets the app check a password. Without it (local runs), files are plain JSON.
"""
from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PBKDF2_ITER = 210_000


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


class Writer:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        for old in self.out.glob("*.json"):
            old.unlink()
        self.key = None
        self.sizes = {}
        pw = os.environ.get("APP_PASSWORD", "")
        if pw:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            salt = os.urandom(16)
            self.key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                                  iterations=PBKDF2_ITER).derive(pw.encode())
            (self.out / "lock.json").write_text(json.dumps({
                "v": 1, "salt": base64.b64encode(salt).decode(), "iter": PBKDF2_ITER,
                "check": self._encrypt(b"nflalgo-ok")}))

    def _encrypt(self, data: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        iv = os.urandom(12)
        ct = AESGCM(self.key).encrypt(iv, data, None)
        return {"iv": base64.b64encode(iv).decode(), "ct": base64.b64encode(ct).decode()}

    def write(self, name, obj):
        raw = json.dumps(clean(obj), separators=(",", ":"))
        if self.key is not None:
            raw = json.dumps(self._encrypt(raw.encode()), separators=(",", ":"))
        p = self.out / name
        p.write_text(raw)
        self.sizes[name] = p.stat().st_size
        return name

    def finish(self):
        """Write files.json: the plain list of file names (no data), used to re-fetch the last
        published copy if a later build of this sport fails."""
        names = sorted(self.sizes) + (["lock.json"] if self.key is not None else [])
        (self.out / "files.json").write_text(json.dumps(names))

    def summary(self):
        total = sum(self.sizes.values())
        return f"{len(self.sizes)} files, {total / 1e6:.1f} MB" + (" (encrypted)" if self.key is not None else "")


def fetch_published(league, name):
    """A file from the last published app, decrypted (None if unavailable). Needs the SITE and
    APP_PASSWORD environment variables (set in the hosted build)."""
    site, pw = os.environ.get("SITE"), os.environ.get("APP_PASSWORD")
    if not site or not pw:
        return None
    import urllib.request
    try:
        get = lambda n: json.loads(urllib.request.urlopen(f"{site}/data/{league}/{n}", timeout=30).read())
        lock, blob = get("lock.json"), get(name)
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=base64.b64decode(lock["salt"]),
                         iterations=int(lock["iter"])).derive(pw.encode())
        return json.loads(AESGCM(key).decrypt(base64.b64decode(blob["iv"]), base64.b64decode(blob["ct"]), None))
    except Exception:  # noqa: BLE001
        return None


def model_params(model, labels, link, switch):
    """The model's formula, so the phone can recompute a game when a starter is switched."""
    sc, rg = model.pipe.named_steps["standardscaler"], model.pipe.named_steps["ridge"]
    p = {"feats": model.feats, "mean": list(sc.mean_), "scale": list(sc.scale_), "coef": list(rg.coef_),
         "intercept": float(rg.intercept_), "labels": [labels.get(f, f) for f in model.feats],
         "link": link, "switch": switch}
    if link == "normal":
        p["sigma"] = model.sigma
    else:
        p["a"], p["b"] = model.a, model.b
    return p
