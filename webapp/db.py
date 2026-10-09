import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app, g
from werkzeug.security import generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    name          TEXT,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- Who gets told about new potholes. A zone (centre + radius) routes reports to the right office;
-- an authority without a zone covers the whole city.
CREATE TABLE IF NOT EXISTS authorities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    department     TEXT,
    email          TEXT,
    phone          TEXT,
    zone_name      TEXT,
    zone_lat       REAL,
    zone_lng       REAL,
    zone_radius_km REAL,
    notify_email   INTEGER NOT NULL DEFAULT 1,
    notify_sms     INTEGER NOT NULL DEFAULT 1,
    is_active      INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    code             TEXT UNIQUE NOT NULL,
    lat              REAL NOT NULL,
    lng              REAL NOT NULL,
    accuracy_m       REAL,
    location_source  TEXT,
    address          TEXT,
    image_path       TEXT NOT NULL,
    annotated_path   TEXT,
    primary_class    TEXT NOT NULL,
    confidence       REAL,
    severity_score   REAL,
    severity         TEXT NOT NULL,
    detections_json  TEXT,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    report_count     INTEGER NOT NULL DEFAULT 1,
    authority_id     INTEGER REFERENCES authorities(id) ON DELETE SET NULL,
    resolution_note  TEXT,
    resolution_image TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_latlng ON reports(lat, lng);

-- Every citizen who reported the same pothole is told when it is fixed.
CREATE TABLE IF NOT EXISTS subscribers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    name       TEXT,
    email      TEXT,
    phone      TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    status     TEXT NOT NULL,
    note       TEXT,
    actor      TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER REFERENCES reports(id) ON DELETE SET NULL,
    audience   TEXT NOT NULL,   -- authority | citizen | test
    channel    TEXT NOT NULL,   -- email | sms
    recipient  TEXT NOT NULL,
    subject    TEXT,
    body       TEXT,
    status     TEXT NOT NULL,   -- sent | failed | not_configured
    error      TEXT,
    created_at TEXT NOT NULL
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc=None):
    con = g.pop("db", None)
    if con is not None:
        con.close()


def init_db(app):
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    con = connect(app.config["DATABASE"])
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    if not con.execute("SELECT 1 FROM admin_users LIMIT 1").fetchone():
        con.execute(
            "INSERT INTO admin_users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (app.config["ADMIN_EMAIL"].lower(), "Administrator",
             generate_password_hash(app.config["ADMIN_PASSWORD"]), now_iso()),
        )
    con.commit()
    con.close()
    app.teardown_appcontext(close_db)
