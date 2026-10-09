"""Adds demo contacts and reports around VIT Vellore so the dashboard and drive mode have something to show.

    python seed_demo.py

Works without the model or torch. To start over, delete instance/potholes.db and uploads/.
"""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import db as dbm
from app import create_app
from detector import RoadDamageDetector
from severity import LABELS, severity

random.seed(7)
CENTER = (12.9692, 79.1559)  # VIT Vellore main gate area
DEMO = [  # (north m, east m, label, area_ratio, conf, status, hours ago)
    (350, 120, "Katpadi Road, near VIT main gate", 0.11, 0.86, "OPEN", 3),
    (-420, 260, "Gandhi Nagar, Katpadi", 0.05, 0.74, "OPEN", 9),
    (780, -300, "Brahmapuram Road", 0.02, 0.61, "IN_PROGRESS", 30),
    (-900, -650, "Sevoor Road junction", 0.14, 0.91, "OPEN", 1),
    (1500, 900, "Katpadi Railway Station Road", 0.04, 0.68, "RESOLVED", 72),
    (-1800, 1200, "Vellore–Katpadi Road", 0.08, 0.80, "OPEN", 20),
    (2600, -400, "Thiruvalam Road", 0.03, 0.57, "IN_PROGRESS", 50),
]


def offset(lat, lng, north_m, east_m):
    from math import cos, radians
    return lat + north_m / 111_320, lng + east_m / (111_320 * cos(radians(lat)))


def demo_photo(path, area_ratio, seed):
    rnd = random.Random(seed)
    W, H = 960, 720
    im = Image.new("RGB", (W, H), (118, 118, 112))
    d = ImageDraw.Draw(im)
    for _ in range(9000):  # asphalt grain
        x, y = rnd.randrange(W), rnd.randrange(H)
        g = rnd.randint(85, 150)
        d.point((x, y), fill=(g, g, g - 4))
    d.polygon([(W * .47, 0), (W * .53, 0), (W * .56, H), (W * .44, H)], fill=(225, 190, 60))  # lane marking
    w = (area_ratio * W * H * 1.5) ** 0.5
    h = w / 1.5
    cx, cy = W * rnd.uniform(.3, .7), H * rnd.uniform(.5, .75)
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    d.ellipse(box, fill=(48, 44, 40))
    d.ellipse((box[0] + w * .12, box[1] + h * .15, box[2] - w * .1, box[3] - h * .1), fill=(32, 30, 28))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(im)
    d.text((14, H - 28), "Demo photo", fill=(240, 240, 240))
    im.save(path, quality=88)
    return im, [round(v, 1) for v in box]


def main():
    app = create_app()
    up = Path(app.config["UPLOAD_DIR"])
    con = dbm.connect(app.config["DATABASE"])
    if not con.execute("SELECT 1 FROM authorities LIMIT 1").fetchone():
        now = dbm.now_iso()
        con.execute("INSERT INTO authorities (name, department, email, phone, zone_name, zone_lat, zone_lng, zone_radius_km,"
                    " notify_email, notify_sms, is_active, created_at) VALUES (?,?,?,?,?,?,?,?,1,1,1,?)",
                    ("Assistant Engineer, Katpadi", "Vellore Corporation, Roads (demo)", "katpadi.roads@example.com",
                     "+919000000001", "Katpadi", CENTER[0], CENTER[1], 2.0, now))
        con.execute("INSERT INTO authorities (name, department, email, phone, zone_name, zone_lat, zone_lng, zone_radius_km,"
                    " notify_email, notify_sms, is_active, created_at) VALUES (?,?,?,?,NULL,NULL,NULL,NULL,1,1,1,?)",
                    ("Roads Control Room", "Vellore Corporation (demo)", "roads.control@example.com",
                     "+919000000002", now))
        print("added 2 demo authority contacts")
    auth_zone = con.execute("SELECT id FROM authorities WHERE zone_lat IS NOT NULL ORDER BY id LIMIT 1").fetchone()
    auth_city = con.execute("SELECT id FROM authorities WHERE zone_lat IS NULL ORDER BY id LIMIT 1").fetchone()
    added = 0
    for i, (n, e, addr, area, conf, status, hours) in enumerate(DEMO):
        code = f"PTH-DM{'ABCEFGH'[i]}{'KMNPQRS'[i]}{i + 2}{i + 3}"
        if con.execute("SELECT 1 FROM reports WHERE code = ?", (code,)).fetchone():
            continue
        lat, lng = offset(*CENTER, n, e)
        im, box = demo_photo(up / f"{code}.jpg", area, i)
        score, label = severity("pothole", area, conf)
        det = {"cls": "pothole", "label": LABELS["pothole"], "confidence": conf, "verifier": None, "box": box,
               "area_ratio": area, "severity_score": score, "severity": label}
        RoadDamageDetector.draw(im, [det]).save(up / f"{code}_detected.jpg", quality=88)
        t = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
        dist_km = ((n ** 2 + e ** 2) ** 0.5) / 1000
        auth = auth_zone if dist_km <= 2.0 else auth_city
        cur = con.execute(
            "INSERT INTO reports (code, lat, lng, accuracy_m, location_source, address, image_path, annotated_path,"
            " primary_class, confidence, severity_score, severity, detections_json, status, report_count, authority_id,"
            " resolution_note, created_at, updated_at, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, lat, lng, 8, "gps", addr, f"{code}.jpg", f"{code}_detected.jpg", "pothole", conf, score, label,
             json.dumps([det]), status, random.choice([1, 1, 2, 3]), auth["id"] if auth else None,
             "Patched with cold-mix asphalt." if status == "RESOLVED" else None, t, t,
             t if status == "RESOLVED" else None))
        con.execute("INSERT INTO report_events (report_id, status, note, actor, created_at) VALUES (?,?,?,?,?)",
                    (cur.lastrowid, "OPEN", "Report received", "citizen", t))
        if status != "OPEN":
            con.execute("INSERT INTO report_events (report_id, status, note, actor, created_at) VALUES (?,?,?,?,?)",
                        (cur.lastrowid, "IN_PROGRESS", None, "Administrator", t))
        if status == "RESOLVED":
            con.execute("INSERT INTO report_events (report_id, status, note, actor, created_at) VALUES (?,?,?,?,?)",
                        (cur.lastrowid, "RESOLVED", "Patched with cold-mix asphalt.", "Administrator", t))
        added += 1
    con.commit()
    con.close()
    print(f"added {added} demo reports around VIT Vellore. Sign in at /admin, or try /drive with Simulate on.")


if __name__ == "__main__":
    main()
