import json
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from severity import LABELS, canonical_class, severity

SEVERITY_COLOURS = {"HIGH": (184, 50, 31), "MEDIUM": (184, 106, 14), "LOW": (44, 116, 72)}
MAX_SIDE = 1600  # photos are stored at most this large


def prepare_photo(stream, out_path):
    """Fix phone rotation, convert to RGB JPEG, shrink very large photos. Returns the original PIL image (with EXIF)."""
    im = Image.open(stream)
    im.load()
    original = im.copy()  # keeps EXIF for the GPS fallback
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im.save(out_path, "JPEG", quality=90)
    return original


class RoadDamageDetector:
    """YOLO26 detector (pothole-only or 4-class) + optional YOLO26-cls verifier for pothole boxes."""

    def __init__(self, cfg):
        self.det_path = Path(cfg["DETECTOR_WEIGHTS"])
        self.ver_path = Path(cfg["VERIFIER_WEIGHTS"])
        deploy = {}
        if Path(cfg["DEPLOY_CONFIG"]).exists():
            deploy = json.loads(Path(cfg["DEPLOY_CONFIG"]).read_text())
        self.conf = float(cfg["DET_CONF"] or deploy.get("det_conf") or 0.30)
        self.ver_thr = float(cfg["VERIFIER_THRESHOLD"] or deploy.get("verifier_threshold") or 0.50)
        self.imgsz = int(cfg["IMGSZ"] or deploy.get("imgsz") or 736)
        self._det = self._ver = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return self.det_path.exists()

    def status(self):
        return {
            "detector": self.det_path.name if self.det_path.exists() else None,
            "verifier": self.ver_path.name if self.ver_path.exists() else None,
            "conf": self.conf, "verifier_threshold": self.ver_thr, "imgsz": self.imgsz,
        }

    def _load(self):
        if self._det is None:
            from ultralytics import YOLO  # imported lazily so the dashboard works without torch installed
            self._det = YOLO(str(self.det_path))
            if self.ver_path.exists():
                self._ver = YOLO(str(self.ver_path))

    def _verify(self, im, box):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        pad_w, pad_h = max(w * 0.2, 16), max(h * 0.2, 16)  # same padding as the training crops
        crop = im.crop((max(0, x0 - pad_w), max(0, y0 - pad_h), min(im.width, x1 + pad_w), min(im.height, y1 + pad_h)))
        r = self._ver.predict(crop, imgsz=224, verbose=False)[0]
        idx = [k for k, v in self._ver.names.items() if v == "pothole"][0]
        return float(r.probs.data[idx])

    def analyze(self, image_path, annotated_path):
        with self._lock:  # one inference at a time keeps memory predictable on a laptop
            self._load()
            im = Image.open(image_path).convert("RGB")
            res = self._det.predict(im, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            W, H = im.size
            detections, rejected = [], 0
            if res.boxes is not None:
                for xyxy, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                    name = canonical_class(self._det.names[int(cls)])
                    ver = None
                    if name == "pothole" and self._ver is not None:
                        ver = self._verify(im, xyxy)
                        if ver < self.ver_thr:
                            rejected += 1
                            continue
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / (W * H)
                    score, label = severity(name, area, conf)
                    detections.append({
                        "cls": name, "label": LABELS.get(name, name.replace("_", " ").title()),
                        "confidence": round(float(conf), 3), "verifier": None if ver is None else round(ver, 3),
                        "box": [round(v, 1) for v in xyxy], "area_ratio": round(area, 4),
                        "severity_score": score, "severity": label,
                    })
        detections.sort(key=lambda d: d["severity_score"], reverse=True)
        self.draw(im, detections).save(annotated_path, "JPEG", quality=88)
        return {"detections": detections, "width": W, "height": H, "rejected_by_verifier": rejected}

    @staticmethod
    def draw(im, detections):
        out = im.copy()
        d = ImageDraw.Draw(out)
        lw = max(3, round(min(out.size) / 220))
        try:
            font = ImageFont.load_default(size=max(14, round(min(out.size) / 38)))
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        for det in reversed(detections):
            colour = SEVERITY_COLOURS[det["severity"]]
            x0, y0, x1, y1 = det["box"]
            d.rectangle([x0, y0, x1, y1], outline=colour, width=lw)
            text = f"{det['label']} {det['confidence']:.2f} | {det['severity'].title()}"
            tb = d.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y0 - th - 2 * lw if y0 - th - 2 * lw > 0 else y0 + lw
            d.rectangle([x0, ty, x0 + tw + 2 * lw, ty + th + 2 * lw], fill=colour)
            d.text((x0 + lw, ty + lw - tb[1]), text, fill=(255, 255, 255), font=font)
        return out
