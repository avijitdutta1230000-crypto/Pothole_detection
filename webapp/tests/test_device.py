"""Pothole box (Raspberry Pi) API test: python tests/test_device.py"""
import io, json, sys, tempfile, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import config, app as appmod, db as dbm

tmp = Path(tempfile.mkdtemp())

# an "old" database without the new columns: the app must add them and keep the rows
old = sqlite3.connect(tmp / "t.db")
old.executescript(dbm.SCHEMA)
old.execute("INSERT INTO reports (code, lat, lng, image_path, primary_class, severity, created_at, updated_at) "
            "VALUES ('PTH-OLD222', 12.9, 79.1, 'x.jpg', 'pothole', 'LOW', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00')")
old.commit(); old.close()


class Cfg(config.Config):
    DATABASE = str(tmp / "t.db"); UPLOAD_DIR = tmp / "up"; TESTING = True
    ADMIN_EMAIL = "admin@test.in"; ADMIN_PASSWORD = "secret123"; SECRET_KEY = "x"
    SMTP_HOST = SMTP_USER = SMTP_PASSWORD = ""; TWILIO_ACCOUNT_SID = TWILIO_AUTH_TOKEN = TWILIO_FROM_NUMBER = ""
    DEVICE_TOKEN = "box-secret"; REOPEN_AFTER_PASSES = 2; REPASS_SECONDS = 0


class NoDet:
    available = True
    def status(self): return {}
    def analyze(self, *a): raise AssertionError("device reports must not run the model again")


class StubGeo:
    def reverse(self, lat, lng): return f"Test Road {lat:.3f}"


a = appmod.create_app(Cfg, detector=NoDet(), geocoder=StubGeo())
c = a.test_client()
notifier = a.extensions["notifier"]
ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or (None if cond else sys.exit(1))
H = {"X-Device-Token": "box-secret"}


def flush():
    notifier.pool.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor
    notifier.pool = ThreadPoolExecutor(max_workers=2)


def notes(**where):
    con = dbm.connect(Cfg.DATABASE)
    rows = [dict(r) for r in con.execute("SELECT * FROM notifications ORDER BY id")]
    con.close()
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def frame():
    b = io.BytesIO(); Image.new("RGB", (1280, 720), (110, 110, 105)).save(b, "JPEG"); b.seek(0); return b


def send(lat, lng, kind="camera", dets=None, jolt=None, headers=H):
    data = {"photo": (frame(), "f.jpg"), "lat": str(lat), "lng": str(lng), "kind": kind, "device_id": "box-01",
            "speed_kmh": "28", "heading": "10",
            "detections": json.dumps(dets if dets is not None else
                                     [{"cls": "pothole", "confidence": 0.8, "verifier": 0.9, "box": [500, 450, 760, 560]}])}
    if jolt:
        data.update(jolt_g=str(jolt), jolt_limit_g="0.67")
    r = c.post("/api/device/report", data=data, headers=headers, content_type="multipart/form-data")
    return r.status_code, r.get_json()


con = dbm.connect(Cfg.DATABASE)
cols = {r["name"] for r in con.execute("PRAGMA table_info(reports)")}
ok({"source", "device_id", "jolt_g", "recheck_count"} <= cols, "old database upgraded with the new columns")
ok(con.execute("SELECT 1 FROM reports WHERE code='PTH-OLD222'").fetchone() is not None, "old reports kept")
con.close()

c.post("/admin/login", data={"email": "admin@test.in", "password": "secret123"})
c.post("/admin/authorities", data={"name": "AE Katpadi", "email": "ae@katpadi.in", "notify_email": "on",
       "zone_name": "Katpadi", "zone_lat": "12.9692", "zone_lng": "79.1559", "zone_radius_km": "2"})

ok(c.get("/api/device/ping").status_code == 401, "ping without token -> 401")
ok(c.get("/api/device/ping", headers=H).get_json()["ok"], "ping with token")
code, j = send(12.9700, 79.1565, headers={"X-Device-Token": "wrong"})
ok(code == 401, "report with wrong token refused")

# camera only
code, j = send(12.9700, 79.1565)
flush()
ok(code == 200 and j["status"] == "created" and j["kind"] == "camera", f"camera report created {j.get('code')}")
first = j["code"]
d = c.get(f"/api/reports/{first}").get_json()["report"]
ok(d["source"] == "device" and d["device_id"] == "box-01" and d["location_source"] == "device", "stored as found by box-01")
ok(len(notes(audience="authority")) == 1, "authority emailed about the new pothole")

# camera + jolt on the same pothole 8 m further on -> merged, jolt raises the severity
code, j = send(12.97007, 79.1565, kind="both", jolt=1.4)
flush()
ok(j["status"] == "merged" and j["code"] == first and j["report_count"] == 2, "camera+jolt pass merged into it")
ok(j["jolt_score"] == 1.0 and j["severity_score"] > d["severity_score"], "jolt raised the severity score")
d2 = c.get(f"/api/reports/{first}").get_json()["report"]
ok(abs(d2["jolt_g"] - 1.4) < 1e-6, "strongest jolt stored on the report")

# jolt only, somewhere new: no detections needed
code, j = send(12.9600, 79.1500, kind="jolt", dets=[], jolt=0.9)
flush()
ok(j["status"] == "created" and j["severity"] in ("MEDIUM", "HIGH"), f"jolt-only report created ({j['severity']})")
code, j = send(12.9600, 79.1500, kind="jolt", dets=[])
ok(code == 400, "jolt report without jolt values refused")
code, j = send(12.9600, 79.1510, kind="camera", dets=[])
ok(code == 400, "camera report without detections refused")

# repair check: mark fixed, then two passes still find it -> reopened, office and reporter told
con = dbm.connect(Cfg.DATABASE); rid = con.execute("SELECT id FROM reports WHERE code=?", (first,)).fetchone()[0]; con.close()
con = dbm.connect(Cfg.DATABASE)
con.execute("INSERT INTO subscribers (report_id, email, created_at) VALUES (?, 'citizen@example.com', ?)", (rid, dbm.now_iso()))
con.commit(); con.close()
r = c.post(f"/api/admin/reports/{rid}/status", data={"status": "RESOLVED", "note": "Patched"})
flush()
ok(r.get_json()["report"]["status"] == "RESOLVED", "office marked it fixed")
h = c.get("/api/hazards?lat=12.9690&lng=79.1565").get_json()
ok(all(x["code"] != first for x in h["hazards"]), "fixed pothole gives no drive warning")

code, j = send(12.97003, 79.15652, kind="both", jolt=1.0)
flush()
ok(j["status"] == "recheck" and j["passes"] == 1 and j["report_status"] == "RESOLVED", "pass 1 still finds it (not yet reopened)")
code, j = send(12.97001, 79.15651, kind="jolt", dets=[], jolt=1.1)
flush()
ok(j["status"] == "reopened" and j["report_status"] == "OPEN", "pass 2 still finds it -> reopened automatically")
re_auth = [n for n in notes(audience="authority") if "Reopened" in (n["subject"] or "")]
re_cit = [n for n in notes(audience="citizen") if "still there" in (n["subject"] or "")]
ok(len(re_auth) == 1 and len(re_cit) == 1, "office and reporter told it is still there")
h = c.get("/api/hazards?lat=12.9690&lng=79.1565").get_json()
ok(any(x["code"] == first for x in h["hazards"]), "reopened pothole warns drivers again")
ev = c.get(f"/api/admin/reports/{rid}").get_json()["report"]["events"]
ok(any("Reopened automatically" in (e["note"] or "") for e in ev), "history shows the automatic reopen")

# device API off when no token is configured
class Off(Cfg):
    DEVICE_TOKEN = ""; DATABASE = str(tmp / "off.db")
off = appmod.create_app(Off, detector=NoDet(), geocoder=StubGeo()).test_client()
ok(off.get("/api/device/ping", headers=H).status_code == 403, "device API off without DEVICE_TOKEN")
print("ALL DEVICE TESTS PASSED")
