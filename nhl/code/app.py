#!/usr/bin/env python3
"""
NHL Algo HUD: a local web app for running the model.

    python app.py            # opens http://127.0.0.1:8766 in your browser
    python app.py --port 9000 --no-browser

Only uses the standard library plus the model's own dependencies
(pandas, numpy, scipy, scikit-learn). Nothing leaves your machine except the
NHL / ESPN downloads when you press "Update data".

For whichever day you look at, the model is refit on every completed game
*before* that day, so past days are honest out-of-sample results.
(Port 8766, so it can run next to the NFL Algo HUD on 8765.)
"""
import argparse
import json
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hud_data as H  # noqa: E402
from paths import REPORTS, ROOT  # noqa: E402

ENGINE = H.ENGINE


def backtest_summary():
    bt = REPORTS / "backtest_results.json"
    if not bt.exists():
        return {"rows": []}
    d = json.loads(bt.read_text())
    return d if isinstance(d, dict) else {"rows": d}


# ---------------------------------------------------------------- background jobs
class Job:
    STEPS = {
        "update": [["update_data.py"]],
        "backtest": [["backtest.py"]],
        "train": [["train.py"]],
        "export": [["train.py"], ["predict.py", "--days", "1"]],
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.name = None
        self.log = []
        self.started = None
        self.finished = None
        self.ok = None

    def snapshot(self):
        with self.lock:
            return {"running": self.running, "name": self.name, "log": self.log[-300:], "ok": self.ok,
                    "started": self.started, "finished": self.finished}

    def start(self, name):
        if name not in self.STEPS:
            raise ValueError(f"unknown job {name}")
        with self.lock:
            if self.running:
                raise RuntimeError("A job is already running.")
            self.running, self.name, self.log, self.ok = True, name, [], None
            self.started, self.finished = datetime.now().isoformat(timespec="seconds"), None
        threading.Thread(target=self._run, args=(name,), daemon=True).start()

    def _run(self, name):
        ok = True
        try:
            for step in self.STEPS[name]:
                self._add(f"$ python {' '.join(step)}")
                proc = subprocess.Popen([sys.executable, "-u", str(HERE / step[0])] + step[1:], cwd=ROOT,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    self._add(line.rstrip())
                ok = proc.wait() == 0
                if not ok:
                    self._add(f"exited with code {proc.returncode}")
                    break
        except Exception as e:  # noqa: BLE001
            ok = False
            self._add(f"{type(e).__name__}: {e}")
        if name == "update" and ok:
            ENGINE.invalidate()
            self._add("rebuilding ratings (a couple of minutes) ...")
            ENGINE.ensure()
            self._add("ready." if ENGINE.error is None else ENGINE.error)
        with self.lock:
            self.running, self.ok = False, ok
            self.finished = datetime.now().isoformat(timespec="seconds")

    def _add(self, line):
        with self.lock:
            self.log.append(line)


JOB = Job()


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (HERE / "hud.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._send(200, H.status(JOB.snapshot()))
            if u.path == "/api/job":
                return self._send(200, JOB.snapshot())
            if not ENGINE.has_data():
                return self._send(409, {"error": "No data yet. Press Update data."})
            if ENGINE.error:
                return self._send(500, {"error": ENGINE.error})
            if u.path == "/api/day":
                ov = dict(x.split(":", 1) for x in q.get("g", "").split(",") if ":" in x)
                return self._send(200, H.day_payload(q["date"], ov))
            if u.path == "/api/record":
                return self._send(200, H.season_record(int(q["season"])))
            if u.path == "/api/rankings":
                return self._send(200, H.rankings())
            if u.path == "/api/players":
                return self._send(200, H.players_payload(q.get("group"), q.get("team"), q.get("q"),
                                                         q.get("sort", "impact"), int(q.get("limit", 80))))
            if u.path == "/api/player":
                return self._send(200, H.player_detail(q["pid"]))
            if u.path == "/api/backtest":
                return self._send(200, backtest_summary())
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/run":
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                JOB.start(body.get("job", ""))
                return self._send(200, JOB.snapshot())
            except (ValueError, RuntimeError) as e:
                return self._send(409, {"error": str(e)})
        return self._send(404, {"error": "not found"})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"NHL Algo HUD running at {url}  (Ctrl+C to stop)")
    threading.Thread(target=ENGINE.ensure, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
