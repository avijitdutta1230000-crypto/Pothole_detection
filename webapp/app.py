import json
import secrets
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, session,
                   url_for)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

import db as dbm
from config import Config
from detector import RoadDamageDetector, caption_bar, prepare_photo
from geo import ReverseGeocoder, bbox, bearing_deg, exif_gps, haversine_m, valid_latlng
from notifier import (Notifier, msg_citizen_received, msg_citizen_reopened, msg_citizen_status, msg_new_report,
                      msg_reopened, normalize_phone, valid_email)
from severity import LABELS, SEVERITY_RANK, canonical_class, fuse, jolt_score, severity

STATUSES = ("OPEN", "IN_PROGRESS", "RESOLVED", "REJECTED")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I


def create_app(config=Config, detector=None, notifier=None, geocoder=None):
    app = Flask(__name__)
    app.config.from_object(config)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    dbm.init_db(app)
    if app.config["ADMIN_PASSWORD"] == "admin123" or app.config["SECRET_KEY"] == "dev-only-change-me":
        print("WARNING: using the default admin password / secret key. Set ADMIN_PASSWORD and SECRET_KEY in .env "
              "before anyone else can reach this server.")

    detector = detector or RoadDamageDetector(app.config)
    notifier = notifier or Notifier(app.config)
    geocoder = geocoder or ReverseGeocoder(app.config["NOMINATIM_USER_AGENT"], app.config["REVERSE_GEOCODE"])
    app.extensions.update(detector=detector, notifier=notifier, geocoder=geocoder)
    base_url = app.config["PUBLIC_BASE_URL"]
    upload_dir = Path(app.config["UPLOAD_DIR"])

    # ------------------------------------------------------------------ helpers
    def err(message, code=400):
        return jsonify({"ok": False, "error": message}), code

    def to_float(v):
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None

    def new_code(con):
        while True:
            code = "PTH-" + "".join(secrets.choice(CODE_CHARS) for _ in range(6))
            if not con.execute("SELECT 1 FROM reports WHERE code = ?", (code,)).fetchone():
                return code

    def upload_url(name):
        return url_for("uploads", filename=name) if name else None

    def report_public(row, con=None):
        d = dict(row)
        d["primary_label"] = LABELS.get(d["primary_class"], d["primary_class"])
        d["image_url"] = upload_url(d["image_path"])
        d["annotated_url"] = upload_url(d["annotated_path"])
        d["resolution_image_url"] = upload_url(d["resolution_image"])
        d["detections"] = json.loads(d.pop("detections_json") or "[]")
        d["track_url"] = f"{base_url}/track/{d['code']}"
        if con is not None:
            d["events"] = [dict(e) for e in con.execute(
                "SELECT status, note, created_at FROM report_events WHERE report_id = ? ORDER BY id", (d["id"],))]
            auth = con.execute("SELECT name, department FROM authorities WHERE id = ?", (d["authority_id"],)).fetchone()
            d["authority"] = dict(auth) if auth else None
        return d

    def add_event(con, report_id, status, note=None, actor=None):
        con.execute("INSERT INTO report_events (report_id, status, note, actor, created_at) VALUES (?, ?, ?, ?, ?)",
                    (report_id, status, note, actor, dbm.now_iso()))

    def route_authority(con, lat, lng):
        """Most specific zone that contains the point; otherwise a city-wide authority."""
        best, best_r, city_wide = None, None, None
        for a in con.execute("SELECT * FROM authorities WHERE is_active = 1 ORDER BY id"):
            if a["zone_lat"] is None or a["zone_radius_km"] is None:
                city_wide = city_wide or a
                continue
            if haversine_m(lat, lng, a["zone_lat"], a["zone_lng"]) <= a["zone_radius_km"] * 1000:
                if best is None or a["zone_radius_km"] < best_r:
                    best, best_r = a, a["zone_radius_km"]
        return best or city_wide

    def authority_recipients(con, authority_id):
        if authority_id:
            rows = con.execute("SELECT * FROM authorities WHERE id = ? AND is_active = 1", (authority_id,)).fetchall()
        else:  # nobody covers this area: tell every active authority
            rows = con.execute("SELECT * FROM authorities WHERE is_active = 1").fetchall()
        return [dict(r) for r in rows]

    def citizen_recipients(con, report_id):
        return [dict(r) for r in con.execute("SELECT email, phone FROM subscribers WHERE report_id = ?", (report_id,))]

    def login_required(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if not session.get("admin_id"):
                if request.path.startswith("/api/"):
                    return err("Sign in again; your session has ended.", 401)
                return redirect(url_for("admin_login", next=request.path))
            return view(*a, **kw)
        return wrapped

    @app.context_processor
    def inject_globals():
        return {"admin_name": session.get("admin_name"), "alert_radius": int(app.config["ALERT_RADIUS_M"])}

    # ------------------------------------------------------------------ pages (citizen)
    @app.get("/")
    def report_page():
        return render_template("report.html", model_ready=detector.available)

    @app.get("/drive")
    def drive_page():
        return render_template("drive.html")

    @app.get("/track")
    def track_lookup():
        code = (request.args.get("code") or "").strip().upper()
        if code and not code.startswith("PTH-"):
            code = "PTH-" + code
        if code and dbm.get_db().execute("SELECT 1 FROM reports WHERE code = ?", (code,)).fetchone():
            return redirect(url_for("track_page", code=code))
        flash("No report with that code. It looks like PTH-7K2M9Q and is in the message we sent you."
              if code else "Enter your report code.")
        return redirect(url_for("report_page"))

    @app.get("/track/<code>")
    def track_page(code):
        con = dbm.get_db()
        row = con.execute("SELECT * FROM reports WHERE code = ?", (code.upper(),)).fetchone()
        if not row:
            abort(404)
        return render_template("track.html", report=report_public(row, con))

    @app.get("/uploads/<path:filename>")
    def uploads(filename):
        return send_from_directory(upload_dir, filename, max_age=3600)

    # ------------------------------------------------------------------ API (citizen)
    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "model": detector.status(), "email": notifier.email_ready,
                        "sms": notifier.sms_ready})

    @app.post("/api/reports")
    def create_report():
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            return err("Add a photo of the road damage.")
        if Path(photo.filename).suffix.lower() not in ALLOWED_EXT:
            return err("Use a JPG, PNG or WebP photo.")

        name = (request.form.get("name") or "").strip()[:80]
        email = (request.form.get("email") or "").strip()[:120]
        phone_raw = (request.form.get("phone") or "").strip()
        if email and not valid_email(email):
            return err("That email address doesn't look right. Check it or leave it empty.")
        phone = normalize_phone(phone_raw, app.config["DEFAULT_COUNTRY_CODE"]) if phone_raw else None
        if phone_raw and not phone:
            return err("That phone number doesn't look right. Use a 10-digit mobile number.")

        if not detector.available:
            return err("The detection model is not installed on the server yet "
                       "(copy pothole_detector.pt into webapp/models/).", 503)

        con = dbm.get_db()
        code = new_code(con)
        img_name, ann_name = f"{code}.jpg", f"{code}_detected.jpg"
        img_path, ann_path = upload_dir / img_name, upload_dir / ann_name
        try:
            original = prepare_photo(photo.stream, img_path)
        except Exception:
            return err("This file could not be opened as a photo. Try taking the picture again.")

        lat, lng = to_float(request.form.get("lat")), to_float(request.form.get("lng"))
        accuracy = to_float(request.form.get("accuracy"))
        source = request.form.get("location_source")
        source = source if source in ("gps", "manual") else "gps"
        if not valid_latlng(lat, lng):
            from_photo = exif_gps(original)
            if from_photo:
                (lat, lng), source, accuracy = from_photo, "photo", None
            else:
                img_path.unlink(missing_ok=True)
                return err("We need the pothole's location. Allow location access, or tap the map to place it.")

        result = detector.analyze(img_path, ann_path)
        dets = result["detections"]
        if not dets:
            img_path.unlink(missing_ok=True)
            ann_path.unlink(missing_ok=True)
            return jsonify({"ok": True, "status": "no_damage",
                            "message": "No road damage found in this photo. Take a closer photo in daylight, "
                                       "with the damage filling more of the frame."})

        top = dets[0]  # highest severity
        address = geocoder.reverse(lat, lng) or f"{lat:.5f}, {lng:.5f}"
        now = dbm.now_iso()

        # Same pothole already reported nearby? Merge instead of creating a duplicate pin.
        radius = app.config["DUPLICATE_RADIUS_M"]
        lo_lat, hi_lat, lo_lng, hi_lng = bbox(lat, lng, radius)
        dup = None
        for r in con.execute("SELECT * FROM reports WHERE status IN ('OPEN','IN_PROGRESS') AND lat BETWEEN ? AND ? "
                             "AND lng BETWEEN ? AND ?", (lo_lat, hi_lat, lo_lng, hi_lng)):
            if haversine_m(lat, lng, r["lat"], r["lng"]) <= radius:
                dup = r
                break

        escalated = False
        if dup:
            escalated = SEVERITY_RANK[top["severity"]] > SEVERITY_RANK[dup["severity"]]
            con.execute("UPDATE reports SET report_count = report_count + 1, updated_at = ? WHERE id = ?",
                        (now, dup["id"]))
            if escalated:
                con.execute("UPDATE reports SET severity = ?, severity_score = ?, primary_class = ? WHERE id = ?",
                            (top["severity"], top["severity_score"], top["cls"], dup["id"]))
            add_event(con, dup["id"], dup["status"], "Reported again by another citizen", "citizen")
            report_id, status_word = dup["id"], "merged"
            img_path.unlink(missing_ok=True)  # the first reporter's photo stays the official one
        else:
            auth = route_authority(con, lat, lng)
            cur = con.execute(
                "INSERT INTO reports (code, lat, lng, accuracy_m, location_source, address, image_path, annotated_path,"
                " primary_class, confidence, severity_score, severity, detections_json, status, authority_id,"
                " created_at, updated_at, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?,?,?,'citizen')",
                (code, lat, lng, accuracy, source, address, img_name, ann_name, top["cls"], top["confidence"],
                 top["severity_score"], top["severity"], json.dumps(dets), auth["id"] if auth else None, now, now))
            report_id, status_word = cur.lastrowid, "created"
            add_event(con, report_id, "OPEN", "Report received", "citizen")

        if email or phone:
            con.execute("INSERT INTO subscribers (report_id, name, email, phone, created_at) VALUES (?, ?, ?, ?, ?)",
                        (report_id, name or None, email or None, phone, now))
        con.commit()

        row = con.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        rep = report_public(row)
        if status_word == "created" or escalated or row["report_count"] in (3, 5, 10):
            subj, text, sms = msg_new_report(rep, base_url, repeat=status_word == "merged")
            notifier.send(report_id, "authority", authority_recipients(con, row["authority_id"]),
                          subj, text, sms, attachment=upload_dir / row["annotated_path"])
        if email or phone:
            subj, text, sms = msg_citizen_received(rep, base_url)
            notifier.send(report_id, "citizen", [{"email": email, "phone": phone}], subj, text, sms)

        return jsonify({
            "ok": True, "status": status_word, "code": rep["code"], "track_url": url_for("track_page", code=rep["code"]),
            "annotated_url": upload_url(ann_name), "address": rep["address"],
            "severity": top["severity"], "severity_score": top["severity_score"], "detections": dets,
            "report_count": rep["report_count"], "location_source": source,
            "rejected_by_verifier": result.get("rejected_by_verifier", 0),
        })

    @app.get("/api/reports/<code>")
    def report_json(code):
        con = dbm.get_db()
        row = con.execute("SELECT * FROM reports WHERE code = ?", (code.upper(),)).fetchone()
        if not row:
            return err("No report with that code.", 404)
        return jsonify({"ok": True, "report": report_public(row, con)})

    @app.get("/api/hazards")
    def hazards():
        """Open potholes around a point, nearest first. Drive mode calls this as the phone moves."""
        lat, lng = to_float(request.args.get("lat")), to_float(request.args.get("lng"))
        if not valid_latlng(lat, lng):
            return err("lat and lng are required.")
        radius = min(to_float(request.args.get("radius")) or app.config["ALERT_RADIUS_M"], 5000)
        include_all = request.args.get("all") == "1"
        lo_lat, hi_lat, lo_lng, hi_lng = bbox(lat, lng, radius)
        out = []
        for r in dbm.get_db().execute(
                "SELECT * FROM reports WHERE status IN ('OPEN','IN_PROGRESS') AND lat BETWEEN ? AND ? "
                "AND lng BETWEEN ? AND ?", (lo_lat, hi_lat, lo_lng, hi_lng)):
            if not include_all and r["primary_class"] != "pothole" and r["severity"] != "HIGH":
                continue  # minor cracks are not worth a driving alert
            dist = haversine_m(lat, lng, r["lat"], r["lng"])
            if dist <= radius:
                out.append({"code": r["code"], "lat": r["lat"], "lng": r["lng"], "distance_m": round(dist, 1),
                            "bearing": round(bearing_deg(lat, lng, r["lat"], r["lng"]), 1),
                            "severity": r["severity"], "type": r["primary_class"],
                            "label": LABELS.get(r["primary_class"], r["primary_class"]),
                            "status": r["status"], "address": r["address"], "report_count": r["report_count"]})
        out.sort(key=lambda h: h["distance_m"])
        return jsonify({"ok": True, "radius_m": radius, "hazards": out})

    # ------------------------------------------------------------------ API (Raspberry Pi pothole box)
    def device_auth():
        """None when the request carries the right device token, otherwise the error response."""
        token = app.config.get("DEVICE_TOKEN") or ""
        if not token:
            return err("The device API is switched off. Set DEVICE_TOKEN in webapp/.env and restart the server.", 403)
        if not secrets.compare_digest(token, request.headers.get("X-Device-Token", "")):
            return err("Wrong or missing device token. Use the same DEVICE_TOKEN as in webapp/.env.", 401)
        return None

    def seconds_since(iso):
        try:
            t = datetime.fromisoformat(iso)
            return (datetime.now(timezone.utc) - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds()
        except (TypeError, ValueError):
            return None

    def nearby_reports(con, lat, lng, radius, statuses):
        lo_lat, hi_lat, lo_lng, hi_lng = bbox(lat, lng, radius)
        marks = ",".join("?" * len(statuses))
        rows = []
        for r in con.execute(f"SELECT * FROM reports WHERE status IN ({marks}) AND lat BETWEEN ? AND ? "
                             "AND lng BETWEEN ? AND ?", (*statuses, lo_lat, hi_lat, lo_lng, hi_lng)):
            d = haversine_m(lat, lng, r["lat"], r["lng"])
            if d <= radius:
                rows.append((d, r))
        return [r for _, r in sorted(rows, key=lambda x: x[0])]

    @app.get("/api/device/ping")
    def device_ping():
        bad = device_auth()
        if bad:
            return bad
        return jsonify({"ok": True, "server_time": dbm.now_iso(), "alert_radius_m": app.config["ALERT_RADIUS_M"],
                        "duplicate_radius_m": app.config["DUPLICATE_RADIUS_M"]})

    @app.post("/api/device/report")
    def device_report():
        """A pothole box saw a pothole (camera), felt one (jolt), or both.

        Header X-Device-Token. Form: device_id, lat, lng, accuracy, speed_kmh, heading, kind (camera|jolt|both),
        detections (JSON list of {cls, confidence, verifier, box: [x0, y0, x1, y1]} in photo pixels), jolt_g,
        jolt_limit_g. File: photo (the camera frame). The box has already run the detector and the verifier,
        so the server does not run the model again.
        """
        bad = device_auth()
        if bad:
            return bad
        f = request.form
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            return err("Send the camera frame as 'photo'.")
        lat, lng = to_float(f.get("lat")), to_float(f.get("lng"))
        if not valid_latlng(lat, lng):
            return err("lat and lng are required (the box has no GPS position yet).")
        kind = f.get("kind") if f.get("kind") in ("camera", "jolt", "both") else "camera"
        device_id = (f.get("device_id") or "box").strip()[:40]
        speed = to_float(f.get("speed_kmh"))
        jolt_g, jolt_limit = to_float(f.get("jolt_g")), to_float(f.get("jolt_limit_g"))
        js = jolt_score(jolt_g, jolt_limit)
        try:
            raw = json.loads(f.get("detections") or "[]")
            if not isinstance(raw, list):
                raise ValueError
        except ValueError:
            return err("detections must be a JSON list.")
        if kind == "jolt" and js is None:
            return err("A jolt report needs jolt_g and jolt_limit_g.")

        con = dbm.get_db()
        code = new_code(con)
        img_name, ann_name = f"{code}.jpg", f"{code}_detected.jpg"
        img_path, ann_path = upload_dir / img_name, upload_dir / ann_name
        try:
            original = prepare_photo(photo.stream, img_path)
        except Exception:
            return err("The photo could not be opened.")
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        scale = W / max(1, original.width)  # boxes are in the pixels of the photo the box sent

        dets = []
        for d in raw[:20]:
            try:
                box = [float(v) * scale for v in d["box"][:4]]
                conf = float(d.get("confidence") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            name = canonical_class(d.get("cls") or "pothole")
            area = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / (W * H)
            photo_score, _ = severity(name, area, conf)
            score, label = fuse(photo_score, js)
            ver = d.get("verifier")
            dets.append({"cls": name, "label": LABELS.get(name, name.replace("_", " ").title()),
                         "confidence": round(conf, 3), "verifier": None if ver is None else round(float(ver), 3),
                         "box": [round(v, 1) for v in box], "area_ratio": round(area, 4),
                         "photo_score": photo_score, "severity_score": score, "severity": label})
        dets.sort(key=lambda d: d["severity_score"], reverse=True)
        if dets:
            top_cls, top_conf = dets[0]["cls"], dets[0]["confidence"]
            score, label = dets[0]["severity_score"], dets[0]["severity"]
        elif js is not None:
            top_cls, top_conf = "pothole", None
            score, label = fuse(None, js)
        else:
            img_path.unlink(missing_ok=True)
            return err("A camera report needs at least one detection.")

        annotated = RoadDamageDetector.draw(im, dets)
        if jolt_g:
            speed_txt = f" at {speed:.0f} km/h" if speed is not None else ""
            annotated = caption_bar(annotated, f"Pothole box {device_id}: jolt {jolt_g:.2f} g{speed_txt}")
        annotated.save(ann_path, "JPEG", quality=88)

        how = {"camera": "seen on camera", "jolt": "felt as a jolt", "both": "seen on camera and felt as a jolt"}[kind]
        actor = f"pothole box {device_id}"
        now = dbm.now_iso()
        radius = app.config["DUPLICATE_RADIUS_M"]
        active = nearby_reports(con, lat, lng, radius, ("OPEN", "IN_PROGRESS"))
        fixed = [r for r in nearby_reports(con, lat, lng, radius, ("RESOLVED",))
                 if (seconds_since(r["resolved_at"]) or 0) <= app.config["REPAIR_CHECK_DAYS"] * 86400]
        escalated, passes, need, still_name = False, None, app.config["REOPEN_AFTER_PASSES"], None

        if active:  # already on the map: count it, and raise the severity if this pass found it worse
            dup = active[0]
            escalated = SEVERITY_RANK[label] > SEVERITY_RANK[dup["severity"]]
            con.execute("UPDATE reports SET report_count = report_count + 1, updated_at = ?, "
                        "jolt_g = CASE WHEN ? IS NULL THEN jolt_g ELSE MAX(COALESCE(jolt_g, 0), ?) END WHERE id = ?",
                        (now, jolt_g, jolt_g, dup["id"]))
            if escalated:
                con.execute("UPDATE reports SET severity = ?, severity_score = ?, primary_class = ? WHERE id = ?",
                            (label, score, top_cls, dup["id"]))
            add_event(con, dup["id"], dup["status"], f"Found again by {actor} ({how})", actor)
            report_id, status_word = dup["id"], "merged"
            img_path.unlink(missing_ok=True)
            ann_path.unlink(missing_ok=True)
        elif fixed:  # marked fixed, but a passing vehicle still finds it: the repair check
            r = fixed[0]
            report_id = r["id"]
            since_last = seconds_since(r["recheck_last"])
            new_pass = since_last is None or since_last >= app.config["REPASS_SECONDS"]
            passes = (r["recheck_count"] or 0) + (1 if new_pass else 0)
            still_name = f"{r['code']}_still_there_{passes}.jpg"
            ann_path.replace(upload_dir / still_name)
            img_path.unlink(missing_ok=True)
            if passes >= need:
                con.execute("UPDATE reports SET status = 'OPEN', resolved_at = NULL, recheck_count = 0, "
                            "recheck_last = NULL, updated_at = ? WHERE id = ?", (now, report_id))
                add_event(con, report_id, "OPEN", f"Reopened automatically: {passes} separate passes still found the "
                                                  "pothole after it was marked fixed", actor)
                status_word = "reopened"
            else:
                con.execute("UPDATE reports SET recheck_count = ?, recheck_last = ? WHERE id = ?",
                            (passes, now if new_pass else r["recheck_last"], report_id))
                if new_pass:
                    add_event(con, report_id, "RESOLVED", f"Still found by {actor} after the repair ({how}); "
                                                          f"pass {passes} of {need} needed to reopen", actor)
                status_word = "recheck"
        else:
            auth = route_authority(con, lat, lng)
            address = geocoder.reverse(lat, lng) or f"{lat:.5f}, {lng:.5f}"
            cur = con.execute(
                "INSERT INTO reports (code, lat, lng, accuracy_m, location_source, address, image_path, annotated_path,"
                " primary_class, confidence, severity_score, severity, detections_json, status, authority_id,"
                " created_at, updated_at, source, device_id, jolt_g, speed_kmh)"
                " VALUES (?,?,?,?,'device',?,?,?,?,?,?,?,?,'OPEN',?,?,?,'device',?,?,?)",
                (code, lat, lng, to_float(f.get("accuracy")), address, img_name, ann_name, top_cls, top_conf, score,
                 label, json.dumps(dets), auth["id"] if auth else None, now, now, device_id, jolt_g, speed))
            report_id, status_word = cur.lastrowid, "created"
            add_event(con, report_id, "OPEN", f"Found by {actor}: {how}", actor)
        con.commit()

        row = con.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        rep = report_public(row)
        if status_word == "created" or (status_word == "merged" and (escalated or row["report_count"] in (3, 5, 10))):
            subj, text, sms = msg_new_report(rep, base_url, repeat=status_word == "merged")
            notifier.send(report_id, "authority", authority_recipients(con, row["authority_id"]), subj, text, sms,
                          attachment=upload_dir / row["annotated_path"])
        elif status_word == "reopened":
            subj, text, sms = msg_reopened(rep, base_url, passes)
            notifier.send(report_id, "authority", authority_recipients(con, row["authority_id"]), subj, text, sms,
                          attachment=upload_dir / still_name)
            subj, text, sms = msg_citizen_reopened(rep, base_url)
            notifier.send(report_id, "citizen", citizen_recipients(con, report_id), subj, text, sms)

        return jsonify({"ok": True, "status": status_word, "code": row["code"], "kind": kind,
                        "severity": label, "severity_score": score, "jolt_score": js,
                        "report_status": row["status"], "report_severity": row["severity"],
                        "report_count": row["report_count"], "passes": passes,
                        "passes_needed": need if passes is not None else None})

    # ------------------------------------------------------------------ admin auth
    @app.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            user = dbm.get_db().execute("SELECT * FROM admin_users WHERE email = ?", (email,)).fetchone()
            if user and check_password_hash(user["password_hash"], request.form.get("password") or ""):
                session.clear()
                session["admin_id"], session["admin_name"] = user["id"], user["name"] or user["email"]
                nxt = request.args.get("next") or ""
                return redirect(nxt if nxt.startswith("/admin") else url_for("admin_dashboard"))
            flash("Email or password is incorrect.")
        return render_template("admin_login.html")

    @app.get("/admin/logout")
    def admin_logout():
        session.clear()
        return redirect(url_for("admin_login"))

    @app.route("/admin/account", methods=["GET", "POST"])
    @login_required
    def admin_account():
        con = dbm.get_db()
        if request.method == "POST":
            f = request.form
            me = con.execute("SELECT * FROM admin_users WHERE id = ?", (session["admin_id"],)).fetchone()
            if f.get("action") == "password":
                if not check_password_hash(me["password_hash"], f.get("current") or ""):
                    flash("Your current password is incorrect.")
                elif len(f.get("new") or "") < 8:
                    flash("Use at least 8 characters for the new password.")
                else:
                    con.execute("UPDATE admin_users SET password_hash = ? WHERE id = ?",
                                (generate_password_hash(f["new"]), me["id"]))
                    con.commit()
                    flash("Password changed.")
            elif f.get("action") == "add":
                email = (f.get("email") or "").strip().lower()
                if not valid_email(email):
                    flash("That email address doesn't look right.")
                elif len(f.get("password") or "") < 8:
                    flash("Use at least 8 characters for their password.")
                elif con.execute("SELECT 1 FROM admin_users WHERE email = ?", (email,)).fetchone():
                    flash("That email already has an account.")
                else:
                    con.execute("INSERT INTO admin_users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                                (email, (f.get("name") or "").strip() or None,
                                 generate_password_hash(f["password"]), dbm.now_iso()))
                    con.commit()
                    flash(f"Account created for {email}. Give them the password you chose.")
            return redirect(url_for("admin_account"))
        users = [dict(u) for u in con.execute("SELECT email, name, created_at FROM admin_users ORDER BY id")]
        return render_template("admin_account.html", users=users)

    # ------------------------------------------------------------------ admin pages
    @app.get("/admin")
    @login_required
    def admin_dashboard():
        con = dbm.get_db()
        has_authority = con.execute("SELECT 1 FROM authorities WHERE is_active = 1 LIMIT 1").fetchone() is not None
        return render_template("admin_dashboard.html", has_authority=has_authority,
                               email_ready=notifier.email_ready, sms_ready=notifier.sms_ready)

    @app.get("/api/admin/reports")
    @login_required
    def admin_reports():
        q = ("SELECT r.*, a.name AS authority_name FROM reports r LEFT JOIN authorities a ON a.id = r.authority_id "
             "WHERE 1=1")
        args = []
        status, sev = request.args.get("status"), request.args.get("severity")
        text = (request.args.get("q") or "").strip()
        if status in STATUSES:
            q += " AND r.status = ?"
            args.append(status)
        elif status == "ACTIVE":
            q += " AND r.status IN ('OPEN','IN_PROGRESS')"
        if sev in SEVERITY_RANK:
            q += " AND r.severity = ?"
            args.append(sev)
        if text:
            q += " AND (r.code LIKE ? OR r.address LIKE ?)"
            args += [f"%{text}%"] * 2
        q += (" ORDER BY CASE r.status WHEN 'OPEN' THEN 0 WHEN 'IN_PROGRESS' THEN 1 ELSE 2 END,"
              " r.severity_score DESC, r.created_at DESC LIMIT 1000")
        con = dbm.get_db()
        rows = []
        for r in con.execute(q, args):
            d = report_public(r)
            d["authority_name"] = r["authority_name"]
            rows.append(d)
        counts = {s: c for s, c in con.execute("SELECT status, COUNT(*) FROM reports GROUP BY status")}
        return jsonify({"ok": True, "reports": rows, "counts": counts})

    @app.get("/api/admin/reports/<int:rid>")
    @login_required
    def admin_report_detail(rid):
        con = dbm.get_db()
        row = con.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        if not row:
            return err("Report not found.", 404)
        d = report_public(row, con)
        d["subscriber_count"] = con.execute("SELECT COUNT(*) FROM subscribers WHERE report_id = ?", (rid,)).fetchone()[0]
        d["notifications"] = [dict(n) for n in con.execute(
            "SELECT audience, channel, recipient, status, error, created_at FROM notifications WHERE report_id = ? "
            "ORDER BY id DESC LIMIT 30", (rid,))]
        return jsonify({"ok": True, "report": d})

    @app.post("/api/admin/reports/<int:rid>/status")
    @login_required
    def admin_update_status(rid):
        status = request.form.get("status")
        note = (request.form.get("note") or "").strip()[:500]
        if status not in STATUSES:
            return err("Unknown status.")
        con = dbm.get_db()
        row = con.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        if not row:
            return err("Report not found.", 404)
        if row["status"] == status:
            return err(f"This report is already {status.replace('_', ' ').lower()}.")
        if status == "REJECTED" and not note:
            return err("Add a short reason so the reporter knows why it was closed.")

        now = dbm.now_iso()
        after_name = None
        after = request.files.get("after_photo")
        if after and after.filename:
            if Path(after.filename).suffix.lower() not in ALLOWED_EXT:
                return err("Use a JPG, PNG or WebP photo for the repair photo.")
            after_name = f"{row['code']}_repaired.jpg"
            try:
                prepare_photo(after.stream, upload_dir / after_name)
            except Exception:
                return err("The repair photo could not be opened.")

        fields = {"status": status, "updated_at": now}
        if status == "RESOLVED":
            fields.update(resolved_at=now, resolution_note=note or None)
            if after_name:
                fields["resolution_image"] = after_name
        elif status in ("OPEN", "IN_PROGRESS"):
            fields["resolved_at"] = None
        if status == "REJECTED":
            fields["resolution_note"] = note
        con.execute(f"UPDATE reports SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                    (*fields.values(), rid))
        add_event(con, rid, status, note or None, session.get("admin_name"))
        con.commit()

        row = con.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        rep = report_public(row, con)
        sent = 0
        if status in ("IN_PROGRESS", "RESOLVED", "REJECTED"):
            auth_name = rep["authority"]["name"] if rep.get("authority") else None
            subj, text, sms = msg_citizen_status(rep, status, note, auth_name, base_url)
            attachment = upload_dir / after_name if after_name else None
            sent = notifier.send(rid, "citizen", citizen_recipients(con, rid), subj, text, sms, attachment=attachment)
        return jsonify({"ok": True, "report": rep, "notified": sent})

    @app.route("/admin/authorities", methods=["GET", "POST"])
    @login_required
    def admin_authorities():
        con = dbm.get_db()
        if request.method == "POST":
            f = request.form
            name = (f.get("name") or "").strip()
            email = (f.get("email") or "").strip()
            phone_raw = (f.get("phone") or "").strip()
            phone = normalize_phone(phone_raw, app.config["DEFAULT_COUNTRY_CODE"]) if phone_raw else None
            zlat, zlng, zr = to_float(f.get("zone_lat")), to_float(f.get("zone_lng")), to_float(f.get("zone_radius_km"))
            problems = []
            if not name:
                problems.append("Add the office or officer name.")
            if not email and not phone_raw:
                problems.append("Add an email address or a phone number; notifications need at least one.")
            if email and not valid_email(email):
                problems.append("The email address doesn't look right.")
            if phone_raw and not phone:
                problems.append("The phone number doesn't look right. Use a 10-digit mobile number.")
            if (zlat is None) != (zlng is None) or (zlat is not None and not zr):
                problems.append("For a zone, pick a centre on the map and enter a radius, or leave all three empty.")
            if problems:
                for p in problems:
                    flash(p)
            else:
                con.execute(
                    "INSERT INTO authorities (name, department, email, phone, zone_name, zone_lat, zone_lng,"
                    " zone_radius_km, notify_email, notify_sms, is_active, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,1,?)",
                    (name, (f.get("department") or "").strip() or None, email or None, phone,
                     (f.get("zone_name") or "").strip() or None, zlat, zlng, zr if zlat is not None else None,
                     1 if f.get("notify_email") else 0, 1 if f.get("notify_sms") else 0, dbm.now_iso()))
                con.commit()
                flash(f"Added {name}. New reports in their area will be sent to them.")
                return redirect(url_for("admin_authorities"))
        rows = con.execute("SELECT a.*, (SELECT COUNT(*) FROM reports r WHERE r.authority_id = a.id AND r.status IN "
                           "('OPEN','IN_PROGRESS')) AS open_count FROM authorities a ORDER BY a.is_active DESC, a.name")
        authorities = [dict(r) for r in rows]
        zones = [{"name": a["zone_name"] or a["name"], "lat": a["zone_lat"], "lng": a["zone_lng"],
                  "radius_km": a["zone_radius_km"], "active": bool(a["is_active"])}
                 for a in authorities if a["zone_lat"] is not None]
        return render_template("admin_authorities.html", authorities=authorities, zones=zones,
                               email_ready=notifier.email_ready, sms_ready=notifier.sms_ready)

    @app.post("/admin/authorities/<int:aid>/toggle")
    @login_required
    def admin_authority_toggle(aid):
        con = dbm.get_db()
        con.execute("UPDATE authorities SET is_active = 1 - is_active WHERE id = ?", (aid,))
        con.commit()
        return redirect(url_for("admin_authorities"))

    @app.post("/admin/authorities/<int:aid>/delete")
    @login_required
    def admin_authority_delete(aid):
        con = dbm.get_db()
        con.execute("DELETE FROM authorities WHERE id = ?", (aid,))
        con.commit()
        flash("Contact removed.")
        return redirect(url_for("admin_authorities"))

    @app.post("/admin/authorities/<int:aid>/test")
    @login_required
    def admin_authority_test(aid):
        con = dbm.get_db()
        a = con.execute("SELECT * FROM authorities WHERE id = ?", (aid,)).fetchone()
        if not a:
            abort(404)
        n = notifier.send(None, "test", [dict(a)], "PotholeWatch test message",
                          "This is a test from PotholeWatch. Pothole reports for your area will arrive like this.",
                          "PotholeWatch test: pothole reports for your area will arrive like this.", wait=True)
        flash(f"Sent {n} test message(s) to {a['name']}. Check the Notifications page for the result.")
        return redirect(url_for("admin_authorities"))

    @app.get("/admin/notifications")
    @login_required
    def admin_notifications():
        rows = dbm.get_db().execute(
            "SELECT n.*, r.code FROM notifications n LEFT JOIN reports r ON r.id = n.report_id "
            "ORDER BY n.id DESC LIMIT 300").fetchall()
        return render_template("admin_notifications.html", rows=[dict(r) for r in rows],
                               email_ready=notifier.email_ready, sms_ready=notifier.sms_ready)

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return err("Not found.", 404)
        return render_template("not_found.html"), 404

    @app.errorhandler(413)
    def too_large(_e):
        return err("That photo is larger than 15 MB. Send a smaller one.", 413)

    return app


if __name__ == "__main__":
    import os
    import sys
    # Phones only share GPS with https pages. `python app.py --https` serves a self-signed certificate
    # (needs `pip install pyopenssl`); accept the browser warning once on the phone.
    ssl_ctx = "adhoc" if "--https" in sys.argv else None
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1",
                     ssl_context=ssl_ctx)
