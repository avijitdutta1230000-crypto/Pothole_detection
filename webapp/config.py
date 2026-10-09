import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _path(name, default):
    """Paths in .env may be relative; they are relative to the webapp folder, not to where you start Python."""
    p = Path(os.getenv(name) or default)
    return str(p if p.is_absolute() else BASE_DIR / p)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    DATABASE = _path("DATABASE", "instance/potholes.db")
    UPLOAD_DIR = Path(_path("UPLOAD_DIR", "uploads"))
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB per upload
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000").rstrip("/")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # Models exported by the Kaggle notebook (unzip pothole_models.zip into webapp/models/)
    DETECTOR_WEIGHTS = _path("DETECTOR_WEIGHTS", "models/pothole_detector.pt")
    VERIFIER_WEIGHTS = _path("VERIFIER_WEIGHTS", "models/pothole_verifier.pt")
    DEPLOY_CONFIG = _path("DEPLOY_CONFIG", "models/deploy_config.json")
    DET_CONF = os.getenv("DET_CONF")                      # overrides deploy_config.json when set
    VERIFIER_THRESHOLD = os.getenv("VERIFIER_THRESHOLD")  # overrides deploy_config.json when set
    IMGSZ = os.getenv("IMGSZ")

    # GPS hazard alerts and duplicate merging
    ALERT_RADIUS_M = _float("ALERT_RADIUS_M", 1000)
    DUPLICATE_RADIUS_M = _float("DUPLICATE_RADIUS_M", 15)

    # First admin login (created on first run)
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

    # Email (Gmail: smtp.gmail.com, port 587, an App Password, not your normal password)
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or 587)
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").replace(" ", "")  # Gmail shows app passwords with spaces
    SMTP_FROM = os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USER", "")
    SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"

    # SMS through Twilio (trial accounts can only text numbers you verified in the Twilio console)
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
    DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "+91")

    NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "PotholeWatch/1.0 (VIT student project)")
    REVERSE_GEOCODE = os.getenv("REVERSE_GEOCODE", "true").lower() == "true"
