import math
import threading
import time

import requests

EARTH_R = 6_371_000.0


def haversine_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bbox(lat, lng, radius_m):
    """Lat/lng box that contains the circle: a cheap SQL prefilter before the exact distance check."""
    dlat = math.degrees(radius_m / EARTH_R)
    dlng = math.degrees(radius_m / (EARTH_R * max(0.01, math.cos(math.radians(lat)))))
    return lat - dlat, lat + dlat, lng - dlng, lng + dlng


def valid_latlng(lat, lng):
    return lat is not None and lng is not None and -90 <= lat <= 90 and -180 <= lng <= 180


class ReverseGeocoder:
    """OpenStreetMap Nominatim, max 1 request/second as its usage policy requires, with a small cache."""

    URL = "https://nominatim.openstreetmap.org/reverse"

    def __init__(self, user_agent, enabled=True):
        self.user_agent, self.enabled = user_agent, enabled
        self._cache, self._lock, self._last = {}, threading.Lock(), 0.0

    def reverse(self, lat, lng):
        if not self.enabled:
            return None
        key = (round(lat, 4), round(lng, 4))  # ~11 m
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            wait = 1.0 - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.get(
                    self.URL,
                    params={"format": "jsonv2", "lat": lat, "lon": lng, "zoom": 18, "addressdetails": 1},
                    headers={"User-Agent": self.user_agent, "Accept-Language": "en"},
                    timeout=6,
                )
                r.raise_for_status()
                data = r.json()
            except Exception:
                return None
        addr = data.get("address", {})
        parts = [
            addr.get("road") or addr.get("pedestrian") or addr.get("neighbourhood"),
            addr.get("suburb") or addr.get("village") or addr.get("town"),
            addr.get("city") or addr.get("state_district") or addr.get("county"),
        ]
        short = ", ".join(dict.fromkeys(p for p in parts if p)) or data.get("display_name")
        self._cache[key] = short
        return short


def exif_gps(pil_image):
    """GPS position stored in a phone photo, or None. Used when the browser cannot share location."""
    try:
        gps = pil_image.getexif().get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def to_deg(v):
            d, m, s = (float(x) for x in v)
            return d + m / 60 + s / 3600

        lat, lng = to_deg(gps[2]), to_deg(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lng = -lng
        return (lat, lng) if valid_latlng(lat, lng) and (lat, lng) != (0.0, 0.0) else None
    except Exception:
        return None
