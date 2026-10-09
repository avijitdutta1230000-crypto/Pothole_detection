"""End-to-end flow test: python tests/test_flows.py  (stub detector + stub geocoder, real notifier without credentials)."""
import io, json, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1]))
from PIL import Image
import config, app as appmod, db as dbm
from detector import RoadDamageDetector

tmp = Path(tempfile.mkdtemp())
class Cfg(config.Config):
    DATABASE = str(tmp / "t.db"); UPLOAD_DIR = tmp / "up"; TESTING = True
    ADMIN_EMAIL = "admin@test.in"; ADMIN_PASSWORD = "secret123"; SECRET_KEY = "x"
    SMTP_HOST = SMTP_USER = SMTP_PASSWORD = ""; TWILIO_ACCOUNT_SID = TWILIO_AUTH_TOKEN = TWILIO_FROM_NUMBER = ""

class StubDet:
    available = True
    def __init__(self): self.next = "pothole"
    def status(self): return {"stub": True}
    def analyze(self, img, ann):
        im = Image.open(img).convert("RGB"); W, H = im.size
        if self.next == "none":
            im.save(ann); return {"detections": [], "width": W, "height": H, "rejected_by_verifier": 1}
        from severity import severity
        area = 0.12 if self.next == "big" else 0.03
        s, lab = severity("pothole", area, 0.85)
        d = {"cls": "pothole", "label": "Pothole", "confidence": 0.85, "verifier": 0.9, "box": [10, 10, 100, 80],
             "area_ratio": area, "severity_score": s, "severity": lab}
        RoadDamageDetector.draw(im, [d]).save(ann)
        return {"detections": [d], "width": W, "height": H, "rejected_by_verifier": 0}

class StubGeo:
    def reverse(self, lat, lng): return f"Test Road {lat:.3f}"

det = StubDet()
a = appmod.create_app(Cfg, detector=det, geocoder=StubGeo())
c = a.test_client()
notifier = a.extensions["notifier"]

def photo(exif_gps=None):
    b = io.BytesIO(); im = Image.new("RGB", (640, 480), (120, 120, 120))
    if exif_gps:
        ex = Image.Exif(); gps = {1: "N", 2: (12.0, 58.0, 9.12), 3: "E", 4: (79.0, 9.0, 21.24)}
        ex[0x8825] = gps; im.save(b, "JPEG", exif=ex)
    else:
        im.save(b, "JPEG")
    b.seek(0); return b

def flush():
    notifier.pool.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor
    notifier.pool = ThreadPoolExecutor(max_workers=2)

def notes(**where):
    con = dbm.connect(Cfg.DATABASE)
    rows = [dict(r) for r in con.execute("SELECT * FROM notifications ORDER BY id")]
    con.close()
    return [r for r in rows if all(r[k] == v for k, v in where.items())]

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or (None if cond else sys.exit(1))

# pages
ok(c.get("/").status_code == 200, "report page")
ok(c.get("/drive").status_code == 200, "drive page")
ok(c.get("/admin").status_code == 302, "admin needs login")
ok(c.get("/api/admin/reports").status_code == 401, "admin API needs login")
r = c.post("/admin/login", data={"email": "admin@test.in", "password": "wrong"})
ok(b"incorrect" in r.data, "wrong password rejected")
r = c.post("/admin/login", data={"email": "ADMIN@test.in", "password": "secret123"})
ok(r.status_code == 302, "login ok")

# authorities: zone around VIT (2 km) + city-wide
r = c.post("/admin/authorities", data={"name": "AE Katpadi", "email": "ae@katpadi.in", "phone": "98765 43210",
           "notify_email": "on", "notify_sms": "on", "zone_name": "Katpadi", "zone_lat": "12.9692", "zone_lng": "79.1559",
           "zone_radius_km": "2"}, follow_redirects=True)
ok(b"Added AE Katpadi" in r.data and b"+919876543210" in r.data, "zone contact added, phone normalised")
r = c.post("/admin/authorities", data={"name": "City control room", "email": "control@city.in", "notify_email": "on"},
           follow_redirects=True)
ok(b"Added City control room" in r.data, "city-wide contact added")
r = c.post("/admin/authorities", data={"name": "Bad", "email": "nope", "phone": "12"}, follow_redirects=True)
ok(b"doesn&#39;t look right" in r.data, "bad contact rejected")

# citizen report inside the Katpadi zone
cit = c  # same client is fine for API
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg"), "lat": "12.9700", "lng": "79.1565", "accuracy": "6",
           "location_source": "gps", "name": "Guddu", "email": "citizen@example.com", "phone": "9123456789"},
           content_type="multipart/form-data")
j = r.get_json(); ok(j["ok"] and j["status"] == "created", f"report created {j.get('code')}")
code = j["code"]; flush()
an = notes(audience="authority")
ok({(n["channel"], n["recipient"]) for n in an} == {("email", "ae@katpadi.in"), ("sms", "+919876543210")},
   "new report sent to the Katpadi zone contact only (email + SMS)")
ok(all(n["status"] == "not_configured" for n in an), "logged as not_configured without credentials")
ok(f"/admin?report={code}" in an[0]["body"] or f"/admin?report={code}" in an[1]["body"], "authority message links to dashboard")
cn = notes(audience="citizen")
ok(len(cn) == 2 and "received" in cn[0]["subject" if cn[0]["channel"] == "email" else "body"].lower(), "citizen got 'received' email + SMS")

# duplicate within 15 m from another citizen, bigger -> HIGH escalates
det.next = "big"
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg"), "lat": "12.97005", "lng": "79.15655",
           "email": "second@example.com"}, content_type="multipart/form-data")
j2 = r.get_json(); flush()
ok(j2["status"] == "merged" and j2["code"] == code and j2["report_count"] == 2, "nearby report merged into the same pothole")
ok(j2["severity"] == "HIGH" and len(notes(audience="authority")) == 4, "severity escalated -> authority told again")

# report outside the zone goes to the city-wide contact
det.next = "pothole"
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg"), "lat": "12.9300", "lng": "79.1300"},
           content_type="multipart/form-data")
j3 = r.get_json(); flush()
last = notes(audience="authority")[-1]
ok(j3["status"] == "created" and last["recipient"] == "control@city.in", "outside every zone -> city-wide contact")

# no location -> EXIF fallback; none at all -> error
r = c.post("/api/reports", data={"photo": (photo(exif_gps=True), "p.jpg")}, content_type="multipart/form-data")
j4 = r.get_json(); ok(j4["ok"] and j4["location_source"] == "photo", "GPS read from photo EXIF when browser has none")
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg")}, content_type="multipart/form-data")
ok(r.status_code == 400 and "location" in r.get_json()["error"], "no location -> clear error")
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg"), "lat": "12.9", "lng": "79.1", "email": "bad@"},
           content_type="multipart/form-data")
ok(r.status_code == 400, "bad email rejected")
det.next = "none"
r = c.post("/api/reports", data={"photo": (photo(), "p.jpg"), "lat": "12.9", "lng": "79.1"}, content_type="multipart/form-data")
ok(r.get_json()["status"] == "no_damage", "photo without damage -> no report")
det.next = "pothole"

# hazards within 1 km for drive mode
h = c.get("/api/hazards?lat=12.9650&lng=79.1565").get_json()   # ~560 m south of the pothole
ok(any(x["code"] == code for x in h["hazards"]), "pothole 560 m away is in the 1 km hazard list")
hz = [x for x in h["hazards"] if x["code"] == code][0]
ok(500 < hz["distance_m"] < 620 and abs(hz["bearing"]) < 15, f"distance {hz['distance_m']} m, bearing {hz['bearing']}° (north)")
h = c.get("/api/hazards?lat=12.9550&lng=79.1565").get_json()   # ~1.67 km away
ok(all(x["code"] != code for x in h["hazards"]), "pothole 1.7 km away is not in the 1 km list")

# authority workflow -> citizen notifications
con = dbm.connect(Cfg.DATABASE); rid = con.execute("SELECT id FROM reports WHERE code=?", (code,)).fetchone()[0]; con.close()
before = len(notes(audience="citizen"))
r = c.post(f"/api/admin/reports/{rid}/status", data={"status": "REJECTED"})
ok(r.status_code == 400, "closing without a reason is refused")
r = c.post(f"/api/admin/reports/{rid}/status", data={"status": "IN_PROGRESS", "note": "Crew on site"})
j = r.get_json(); flush()
ok(j["ok"] and j["notified"] == 3, "repair started -> 3 messages (citizen 1 email+SMS, citizen 2 email)")
r = c.post(f"/api/admin/reports/{rid}/status", data={"status": "RESOLVED", "note": "Filled with hot-mix",
           "after_photo": (photo(), "after.jpg")}, content_type="multipart/form-data")
j = r.get_json(); flush()
fixed = [n for n in notes(audience="citizen")[before:] if n["subject"] and n["subject"].startswith("Fixed")]
ok(j["ok"] and len(fixed) == 2 and "Filled with hot-mix" in fixed[0]["body"], "fixed -> both reporters emailed with the note")
sms_fixed = [n for n in notes(audience="citizen", channel="sms") if "is fixed" in n["body"]]
ok(len(sms_fixed) == 1 and sms_fixed[0]["recipient"] == "+919123456789", "fixed -> SMS to the reporter's phone")
t = c.get(f"/track/{code}")
ok(t.status_code == 200 and b"After repair" in t.data and b"Filled with hot-mix" in t.data, "track page shows fix + after photo")
h = c.get("/api/hazards?lat=12.9650&lng=79.1565").get_json()
ok(all(x["code"] != code for x in h["hazards"]), "fixed pothole no longer triggers GPS alerts")
ok(c.get(f"/track?code={code[4:].lower()}").status_code == 302, "track lookup accepts code without prefix")
ok(c.get("/api/reports/PTH-NOPE22").status_code == 404, "unknown code -> 404 JSON")

# dashboard API
d = c.get("/api/admin/reports?status=RESOLVED").get_json()
ok(len(d["reports"]) == 1 and d["counts"]["RESOLVED"] == 1, "dashboard filter by status")
d = c.get(f"/api/admin/reports/{rid}").get_json()["report"]
ok(d["subscriber_count"] == 2 and len(d["events"]) >= 4, "detail shows subscribers + history")

# notifications page + test send + account
ok(c.get("/admin/notifications").status_code == 200, "notifications page")
r = c.post("/admin/authorities/1/test", follow_redirects=True)
ok(b"Sent 2 test message" in r.data, "test message to a contact")
r = c.post("/admin/account", data={"action": "password", "current": "secret123", "new": "newpass99"}, follow_redirects=True)
ok(b"Password changed" in r.data, "password change")
r = c.post("/admin/account", data={"action": "add", "email": "jen@corp.in", "password": "abcdefgh"}, follow_redirects=True)
ok(b"Account created" in r.data, "staff account added")
c.get("/admin/logout")
ok(c.post("/admin/login", data={"email": "admin@test.in", "password": "newpass99"}).status_code == 302, "login with new password")
print("ALL FLOW TESTS PASSED")
