"""Pothole & road-damage detection: YOLO26 training pipeline (Kaggle script version).
Generated from the notebook; see the notebook for the explanations.
"""

# %%
# # Pothole & road-damage detection: YOLO26 training pipeline (Kaggle)
#
# This notebook trains three models and exports everything the web app needs:
#
# | Track | Model | Task | What you report |
# |---|---|---|---|
# | A | YOLO26 detector | 4 classes: longitudinal, transverse, alligator crack, pothole (RDD2022 India) | P, R, mAP50, mAP50-95 per class |
# | B | YOLO26 detector | pothole only (RDD2022 potholes + any extra pothole dataset you attach) | pothole P, R, mAP50 |
# | C | YOLO26-cls verifier | checks every pothole box: *pothole* vs *not pothole* | crop accuracy, precision, recall |
#
# The last section evaluates the full two-stage system (detector → verifier) on the **held-out India test split** and
# reports box-level precision/recall/F1 and image-level accuracy ("does this photo contain a pothole?").
#
# **Before you run**
# 1. *Settings → Accelerator → GPU T4 x2* and *Settings → Internet → On* (needed to download YOLO26 weights).
# 2. *Add Input* → your RDD2022 dataset. Either works:
#    * your existing 4-class YOLO version (`data.yaml` + `images/` + `labels/`, or folders with a `classes.txt`), or
#    * raw RDD2022 (Pascal VOC XML, `India/train/annotations/xmls`), also as the original per-country zip files.
#      Other countries (Japan, Czech, …) are used as extra training data if present.
#
#    If cell 4 stops with "No labelled RDD2022 India images found", it prints the folders it sees and why each was
#    skipped. Fix the input (or set `MANUAL_DATA_YAML` / `MANUAL_CLASS_NAMES` in cell 2) and run again.
# 3. Optional: *Add Input* → any pothole detection dataset (YOLO or VOC format). It is used for Track B only.
# 4. Run with **Save Version → Save & Run All (Commit)** so training continues if your browser disconnects.
#    Everything ends up in `/kaggle/working/export/` and `pothole_models.zip`.
#
# **YOLO26 or YOLO27?** YOLO26 is the newest Ultralytics model you can train today. YOLO27 has been announced but its
# weights are not published yet. `MODEL_FAMILY` below is set to try YOLO27 first and fall back to YOLO26, so the same
# notebook picks up YOLO27 automatically once it is released.
#
# **About the target numbers (92–95 %).** Published, architecture-modified YOLOv8 models report about 62–63 % mAP50 on
# RDD2022 (RDD-YOLO, *Applied Sciences* 2024: 62.5 %; SEA-YOLO v8: 63.2 %), so a 4-class mAP50 in the 90s is not a
# realistic target and an examiner who knows the dataset will question it. The metrics in this notebook that can
# legitimately reach the 90s are the pothole verifier's crop accuracy and the end-to-end image-level accuracy
# ("does this photo contain a pothole?"). Nothing here is evaluated on training images. Whatever you get, put the
# metric's name next to the number on your slide.

# %%
# 1. Install / check environment
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', *['install', '-q', '-U', 'ultralytics>=8.4.0']], check=True)
import os
os.environ["WANDB_MODE"] = "disabled"  # Kaggle ships wandb; never let it ask for an API key mid-run
import ultralytics, torch, platform
from ultralytics import settings
settings.update({k: False for k in ("wandb", "mlflow", "comet", "clearml", "dvc", "neptune", "raytune")
                 if k in settings})
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| python", platform.python_version())
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
# 2. Configuration: edit here only
from pathlib import Path
import torch

MODEL_FAMILY = "yolo27"   # tried first; falls back to yolo26 (then yolo11) when those weights don't exist yet
DET_SIZE = "s"            # n / s / m / l   (s fits Kaggle time limits; m is ~2x slower, usually +2-4 mAP)
CLS_SIZE = "s"
IMGSZ = 736               # RDD India photos are 720x720; training at 640 throws away thin-crack detail

# Training budget in hours. When TIME_* is set, Ultralytics ignores EPOCHS_* and fits the LR schedule to the time
# (early stopping still applies). 6 + 2.5 + 0.75 h keeps the whole notebook inside one 12 h Kaggle session.
# For a quick smoke test set TIME_* = None and EPOCHS_* = 3.
TIME_A, TIME_B, TIME_C = 6.0, 2.5, 0.75
EPOCHS_A, EPOCHS_B, EPOCHS_C = 200, 150, 60
PATIENCE = 40
SEED = 42

RUN_TRACK_A = True        # 4-class road-damage detector (matches your slides)
RUN_TRACK_B = True        # pothole-only detector (used by the web app and GPS alerts)
RUN_TRACK_C = True        # pothole crop verifier (removes false positives)
TRACK_B_WEIGHTS = None    # set to a best.pt path to run Track C in a later session without retraining B

PRIMARY_COUNTRY = "India"
USE_OTHER_COUNTRIES = True    # add other RDD2022 countries to TRAIN only (val/test stay India)
EXTRA_COUNTRIES = ["Japan", "Czech", "United_States", "China_MotorBike"]  # Norway (huge images) and drone views left out
MAX_EXTRA_IMAGES = 8000       # cap on extra-country images so an epoch stays affordable
BACKGROUND_FRACTION = 0.10    # keep ~10% damage-free training images as negatives
MAX_REPEAT = 3                # repeat-factor oversampling cap for rare classes (transverse cracks are rare in India)
SPLIT = (0.85, 0.10, 0.05)    # used only when the dataset has no split of its own

MANUAL_DATA_YAML = None   # e.g. "/kaggle/input/rdd2022-india-4class/data.yaml" if auto-discovery picks the wrong one
MANUAL_CLASS_NAMES = None # only for YOLO labels with no data.yaml/classes.txt: names in label-id order, e.g.
                          # ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
EXTRACT_ROOT = Path("/tmp/extracted_inputs")  # zip/tar files inside your dataset are unpacked here

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA, RUNS, EXPORT = WORK / "data", WORK / "runs", WORK / "export"
for p in (DATA, RUNS, EXPORT):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
POTHOLE = 3

NGPU = torch.cuda.device_count()
DEVICE = [0, 1] if NGPU > 1 else (0 if NGPU == 1 else "cpu")   # both T4s train together (DDP)
EVAL_DEVICE = 0 if NGPU else "cpu"
BATCH = {"n": 32, "s": 16, "m": 10, "l": 8}[DET_SIZE] * max(1, NGPU)
WORKERS = 4
print("device:", DEVICE, "| batch:", BATCH)

# %%
# 3. Helpers: pretrained weights, paths, labels
import os, json, random, shutil, math, time, yaml, glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image

random.seed(SEED); np.random.seed(SEED)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COUNTRIES = ["India", "Japan", "Czech", "Norway", "United_States", "China_MotorBike", "China_Drone"]
USED_WEIGHTS = {}


def load_pretrained(kind="det", size="s"):
    from ultralytics import YOLO
    suffix = "-cls" if kind == "cls" else ""
    for fam in dict.fromkeys([MODEL_FAMILY, "yolo26", "yolo11"]):
        name = f"{fam}{size}{suffix}.pt"
        try:
            m = YOLO(name)
            print("Pretrained weights:", name)
            USED_WEIGHTS[kind] = name
            return m, name
        except Exception as e:
            print(f"  {name} unavailable ({type(e).__name__}), trying the next family")
    raise RuntimeError("Could not download any pretrained weights. Turn Internet ON in the notebook settings.")


def canon_class(name):
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if n == "d00" or "longitud" in n: return 0
    if n == "d10" or "transvers" in n or "lateral" in n: return 1
    if n == "d20" or "alligator" in n or "crocodile" in n or "fatigue" in n: return 2
    if n == "d40" or "pothole" in n or "pot_hole" in n: return 3
    return None


def country_of(path):
    s, name = str(path).lower(), Path(path).name.lower()
    for c in COUNTRIES:
        if f"/{c.lower()}/" in s or name.startswith(c.lower() + "_"):
            return c
    return None


def dataset_key(path):
    # the dataset a file belongs to: /kaggle/input/<slug>/... (or /kaggle/input/datasets/<owner>/<slug>/...)
    for root in (INPUT_ROOT, EXTRACT_ROOT):
        try:
            parts = Path(path).relative_to(root).parts
        except ValueError:
            continue
        n = 3 if parts and parts[0] in ("datasets", "competitions") and len(parts) > 3 else 1
        return "/".join(parts[:n])
    return str(Path(path).parent)


def label_path_for(img):
    s = str(img); i = s.rfind(f"{os.sep}images{os.sep}")
    if i < 0:
        return None
    return Path(s[:i] + f"{os.sep}labels{os.sep}" + s[i + 8:]).with_suffix(".txt")


def read_yolo_labels(lbl, remap):
    boxes = []
    if lbl is not None and lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = remap.get(int(float(parts[0])))
            if c is None:
                continue
            v = list(map(float, parts[1:]))
            if len(v) > 4:  # polygon label -> box
                xs, ys = v[0::2], v[1::2]
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            cx, cy, w, h = v[:4]
            if w > 0 and h > 0:
                boxes.append((c, cx, cy, w, h))
    return boxes


def image_size(path):
    with Image.open(path) as im:
        return im.size


def xyxy_to_yolo(c, x0, y0, x1, y1, W, H):
    x0, x1 = max(0.0, min(x0, x1)), min(W, max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(H, max(y0, y1))
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return None
    return (c, (x0 + w / 2) / W, (y0 + h / 2) / H, w / W, h / H)


def safe_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.abspath(src), dst)


def write_label(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))


def find_best(run_name):
    hits = sorted(RUNS.glob(f"**/{run_name}*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None

# %%
# 4. Find the datasets under /kaggle/input
#    Reads YOLO (data.yaml, or images/ + labels/ folders), Pascal VOC XML (raw RDD2022), and unpacks zip/tar
#    archives found inside a dataset (the official RDD2022 download has one zip per country).
import re, zipfile, tarfile

ARCHIVE_RE = re.compile(r"\.(zip|tar|tar\.gz|tgz)$", re.I)
SKIP_ARCHIVE_RE = re.compile(r"norway|drone", re.I)   # very large / aerial views; not used for training
NAMES_FILES = ("classes.txt", "obj.names", "_darknet.labels", "labels.txt")
ROOTS = [INPUT_ROOT, EXTRACT_ROOT]


def dataset_root(path):
    for root in ROOTS:
        try:
            Path(path).relative_to(root)
        except ValueError:
            continue
        return root / dataset_key(path)
    return Path(path).parent


def unpack_archives(max_rounds=3):
    for _ in range(max_rounds):
        new = 0
        archives = []
        for root in ROOTS:
            if root.exists():
                for dp, dn, fn in os.walk(root, followlinks=True):
                    archives += [Path(dp) / f for f in fn if ARCHIVE_RE.search(f) and not SKIP_ARCHIVE_RE.search(f)]
        for arc in archives:
            dest = EXTRACT_ROOT / dataset_key(arc) / ARCHIVE_RE.sub("", arc.name)
            if (dest / ".unpacked").exists():
                continue
            try:
                if arc.suffix.lower() == ".zip":
                    with zipfile.ZipFile(arc) as z:
                        need = sum(i.file_size for i in z.infolist())
                else:
                    need = arc.stat().st_size * 2
                free = shutil.disk_usage(EXTRACT_ROOT.parent).free
                if need > free * 0.9:
                    print(f"  skipping {arc.name}: needs {need / 1e9:.1f} GB, only {free / 1e9:.1f} GB free")
                    continue
                print(f"unpacking {arc} ({need / 1e9:.2f} GB) ...", flush=True)
                dest.mkdir(parents=True, exist_ok=True)
                if arc.suffix.lower() == ".zip":
                    with zipfile.ZipFile(arc) as z:
                        z.extractall(dest)
                else:
                    with tarfile.open(arc) as t:
                        try:
                            t.extractall(dest, filter="data")
                        except TypeError:  # Python without extraction filters
                            t.extractall(dest)
                (dest / ".unpacked").touch()
                new += 1
            except Exception as e:
                print(f"  could not unpack {arc.name}: {type(e).__name__}: {e}")
        if not new:
            break


EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
unpack_archives()

t0 = time.time()
img_index, xml_files, yaml_files, label_dirs, other = defaultdict(list), [], [], set(), Counter()
for root in ROOTS:
    for dp, dn, fn in os.walk(root, followlinks=True):
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            p = os.path.join(dp, f)
            if ext in IMG_EXT:
                img_index[(dataset_key(p), os.path.splitext(f)[0])].append(p)
            elif ext == ".xml":
                xml_files.append(p)
            elif ext in (".yaml", ".yml"):
                yaml_files.append(p)
            elif ext == ".txt" and f"{os.sep}labels" in dp and f not in NAMES_FILES:
                label_dirs.add(dp)
            elif ext in (".json", ".csv"):
                other[(dataset_key(p), ext)] += 1
print(f"scanned input in {time.time() - t0:.0f}s: {sum(map(len, img_index.values()))} images, "
      f"{len(xml_files)} xml, {len(yaml_files)} yaml, {len(label_dirs)} YOLO label folders")

SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}
REPORT = []  # what happened to each dataset, printed if nothing usable is found


def resolve_split(yaml_path, d, split):
    ydir, vals, cands = yaml_path.parent, [], []
    for k in [split] + (["valid"] if split == "val" else []):
        v = d.get(k)
        if v:
            vals += v if isinstance(v, list) else [v]
    for v in map(str, vals):
        stripped = v.replace("../", "").lstrip("./")
        if d.get("path"):
            cands.append(Path(str(d["path"])) / v)
        cands += [ydir / v, ydir / stripped, ydir.parent / stripped]
    for a in SPLIT_ALIASES[split]:
        cands += [ydir / "images" / a, ydir / a / "images", ydir / a]
    for c in cands:
        try:
            c = Path(os.path.normpath(str(c)))
            if c.is_dir() and any(c.iterdir()):
                return c
        except Exception:
            pass
    # data.yaml written on another machine (Colab / Windows paths): match the last folders of the path
    base = dataset_root(yaml_path)
    for v in map(str, vals):
        parts = [p for p in re.split(r"[\\/]+", v) if p not in ("", ".", "..")]
        for k in (3, 2):
            if len(parts) >= k:
                hits = sorted(h for h in base.glob("**/" + "/".join(parts[-k:])) if h.is_dir())
                if hits:
                    return hits[0]
    for a in SPLIT_ALIASES[split]:
        hits = sorted(h for pat in (f"**/{a}/images", f"**/images/{a}") for h in base.glob(pat) if h.is_dir())
        if hits:
            return hits[0]
    return None


def names_near(path):
    # class names from classes.txt / obj.names, searched from `path` up to the dataset root
    root, p = dataset_root(path), Path(path)
    for folder in [p, *p.parents]:
        for n in NAMES_FILES:
            f = folder / n
            if f.exists():
                names = [l.strip() for l in f.read_text(errors="ignore").splitlines() if l.strip()]
                if names:
                    return dict(enumerate(names)), str(f)
        for f in sorted(folder.glob("*.y*ml")) if folder.is_dir() else []:
            try:
                nm = (yaml.safe_load(open(f)) or {}).get("names")
            except Exception:
                nm = None
            if nm:
                return (nm if isinstance(nm, dict) else dict(enumerate(nm))), str(f)
        if folder == root or folder == folder.parent:
            break
    if MANUAL_CLASS_NAMES:
        return dict(enumerate(MANUAL_CLASS_NAMES)), "MANUAL_CLASS_NAMES"
    return None, None


def kind_of(names):
    remap = {int(k): canon_class(v) for k, v in names.items()}
    mapped = {v for v in remap.values() if v is not None}
    if len(mapped) >= 3:
        return "rdd", remap
    if mapped == {POTHOLE}:
        return "pothole", remap
    return None, remap


items = []  # each: dict(img, boxes=[(c,cx,cy,w,h)], split, domain, country, source)
seen_imgs = set()


def add_yolo_images(img_paths, remap, kind, split, source):
    n = 0
    for img in img_paths:
        if os.path.splitext(img)[1].lower() not in IMG_EXT or img in seen_imgs:
            continue
        seen_imgs.add(img)
        boxes = read_yolo_labels(label_path_for(img), remap)
        if kind == "rdd":
            ctry = country_of(img) or PRIMARY_COUNTRY
            if ctry != PRIMARY_COUNTRY and not (USE_OTHER_COUNTRIES and ctry in EXTRA_COUNTRIES):
                continue
            dom = "primary" if ctry == PRIMARY_COUNTRY else "extra_country"
            sp = split if dom == "primary" else "train"
        else:
            ctry, dom, sp = None, "pothole_extra", split
        items.append(dict(img=img, boxes=boxes, split=sp, domain=dom, country=ctry, source=source))
        n += 1
    return n


# --- YOLO datasets with a data.yaml
yaml_list = [Path(MANUAL_DATA_YAML)] if MANUAL_DATA_YAML else [Path(p) for p in yaml_files]
for yp in yaml_list:
    try:
        d = yaml.safe_load(open(yp))
    except Exception as e:
        REPORT.append(f"{yp}: unreadable yaml ({type(e).__name__})")
        continue
    if not isinstance(d, dict) or not any(k in d for k in ("train", "val")):
        continue  # not a dataset yaml (configs etc.)
    names = d.get("names")
    src = "data.yaml"
    if not names:
        names, src = names_near(yp)
    if not names:
        REPORT.append(f"{yp}: has train/val paths but no class names (add 'names:' or set MANUAL_CLASS_NAMES)")
        continue
    names = names if isinstance(names, dict) else dict(enumerate(names))
    kind, remap = kind_of(names)
    if kind is None:
        REPORT.append(f"{yp}: skipped, classes {list(names.values())} are not RDD2022 road-damage classes")
        continue
    added, found_split = 0, {}
    for split in ("train", "val", "test"):
        sdir = resolve_split(yp, d, split)
        found_split[split] = str(sdir) if sdir else None
        if sdir is not None:
            added += add_yolo_images(sorted(glob.glob(str(sdir / "**" / "*.*"), recursive=True)), remap, kind,
                                     split, str(yp))
    msg = f"{yp} [{kind}]: {added} images (train={found_split['train']}, val={found_split['val']})"
    print("YOLO dataset", msg)
    REPORT.append(msg if added else msg + "  <- no images: the train/val paths in this yaml don't exist here")

# --- YOLO images/ + labels/ folders without a data.yaml
for ld in sorted(label_dirs):
    parts = Path(ld).parts
    idx = [j for j, p in enumerate(parts) if p == "labels"]
    if not idx:
        continue
    i = idx[-1]
    img_dir = Path(*parts[:i], "images", *parts[i + 1:])
    if not img_dir.is_dir():
        continue
    imgs = [str(p) for p in sorted(img_dir.glob("*")) if p.suffix.lower() in IMG_EXT]
    if not imgs or all(p in seen_imgs for p in imgs[:20]):
        continue  # already read through a data.yaml
    names, src = names_near(ld)
    if not names:
        REPORT.append(f"{ld}: YOLO labels without class names; add classes.txt or set MANUAL_CLASS_NAMES")
        continue
    kind, remap = kind_of(names)
    if kind is None:
        REPORT.append(f"{ld}: skipped, classes {list(names.values())} ({src}) are not road-damage classes")
        continue
    split = next((s for p in reversed(parts) for s, al in SPLIT_ALIASES.items() if p.lower() in al), None)
    n = add_yolo_images(imgs, remap, kind, split, str(ld))
    msg = f"{ld} [{kind}, names from {src}]: {n} images, split={split or 'assigned below'}"
    print("YOLO folder", msg); REPORT.append(msg)

primary_from_yolo = any(it["domain"] == "primary" for it in items)

# --- Pascal VOC XML (raw RDD2022 or any VOC export), grouped per dataset
voc_groups = defaultdict(list)
for xp in xml_files:
    try:
        root = ET.parse(xp).getroot()
    except Exception:
        continue
    if root.tag == "annotation":
        voc_groups[dataset_key(xp)].append((xp, root))

n_voc, seen_stems = Counter(), set()
for key, group in voc_groups.items():
    classes = {canon_class(o.findtext("name")) for _, r in group for o in r.iter("object")} - {None}
    group_kind = "rdd" if len(classes) >= 3 else ("pothole" if classes == {POTHOLE} else None)
    if group_kind is None:
        REPORT.append(f"{key}: {len(group)} VOC xml files skipped, classes are not road damage")
        continue
    matched = 0
    for xp, root in group:
        ctry = country_of(xp) if group_kind == "rdd" else None
        if group_kind == "rdd":
            ctry = ctry or PRIMARY_COUNTRY
            if ctry == PRIMARY_COUNTRY and primary_from_yolo:
                continue  # already have India from the YOLO dataset
            if ctry != PRIMARY_COUNTRY and not (USE_OTHER_COUNTRIES and ctry in EXTRA_COUNTRIES):
                continue
        fname = root.findtext("filename") or (Path(xp).stem + ".jpg")
        cands = img_index.get((key, os.path.splitext(os.path.basename(fname))[0])) or img_index.get((key, Path(xp).stem))
        if not cands:
            continue
        img = max(cands, key=lambda c: len(os.path.commonprefix([c, xp])))
        stem_key = (ctry, Path(img).stem)
        if img in seen_imgs or stem_key in seen_stems:
            continue  # same photo in an extracted copy and a folder copy
        matched += 1
        W = float(root.findtext("size/width") or 0); H = float(root.findtext("size/height") or 0)
        if W <= 0 or H <= 0:
            W, H = image_size(img)
        boxes = []
        for o in root.iter("object"):
            c, bb = canon_class(o.findtext("name")), o.find("bndbox")
            if c is None or bb is None:
                continue
            b = xyxy_to_yolo(c, *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")), W, H)
            if b:
                boxes.append(b)
        if group_kind == "rdd":
            dom = "primary" if ctry == PRIMARY_COUNTRY else "extra_country"
        else:
            dom = "pothole_extra"
        seen_imgs.add(img); seen_stems.add(stem_key)
        items.append(dict(img=img, boxes=boxes, split="train" if dom == "extra_country" else None,
                          domain=dom, country=ctry, source="voc"))
        n_voc[dom] += 1
    REPORT.append(f"{key}: {len(group)} VOC xml [{group_kind}], {matched} matched to images")
print("VOC images added:", dict(n_voc))


def show_tree(path, depth=3, max_lines=60):
    lines = []
    def walk(p, level):
        if level > depth or len(lines) >= max_lines:
            return
        try:
            entries = sorted(p.iterdir())
        except Exception:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if not e.is_dir()]
        exts = Counter(e.suffix.lower() or "(no ext)" for e in files)
        summary = ", ".join(f"{n} {x}" for x, n in exts.most_common(4))
        lines.append("    " * level + f"{p.name}/" + (f"  [{summary}]" if summary else ""))
        for d in dirs[:12]:
            walk(d, level + 1)
        if len(dirs) > 12:
            lines.append("    " * (level + 1) + f"... {len(dirs) - 12} more folders")
    walk(path, 0)
    print("\n".join(lines))


def explain_inputs():
    print("\n" + "=" * 100 + "\nNo labelled India road-damage images were found. Here is what the notebook sees:\n")
    datasets = [p for p in sorted(INPUT_ROOT.iterdir())] if INPUT_ROOT.exists() else []
    if not datasets:
        print(f"{INPUT_ROOT} is EMPTY: no dataset is attached to this notebook.\n"
              "  Fix: right panel -> 'Add Input' -> search for your dataset -> '+'. If you just uploaded it, wait until\n"
              "  Kaggle says it is ready, then add it. For a committed run, attach it before 'Save Version'.")
        return
    for p in datasets:
        show_tree(p)
    if any(EXTRACT_ROOT.iterdir()):
        print("\nunpacked archives:"); show_tree(EXTRACT_ROOT)
    print("\nWhat happened to each dataset:")
    for r in REPORT or ["nothing recognisable (no data.yaml with road-damage classes, no YOLO labels, no VOC xml)"]:
        print("  -", r)
    if other:
        print("  - also found:", {f"{k} {e}": n for (k, e), n in other.items()},
              "(COCO json / CSV annotations are not read by this notebook)")
    print("\nThe notebook accepts:\n"
          "  * raw RDD2022: India/train/images/*.jpg + India/train/annotations/xmls/*.xml (zips are unpacked)\n"
          "  * YOLO: a data.yaml with 'names:' and train/val folders, or images/ + labels/ with classes.txt\n"
          "  If your YOLO data has no class names, set MANUAL_CLASS_NAMES in cell 2 (in label-id order).\n"
          "  If the right data.yaml is listed above but not used, set MANUAL_DATA_YAML to its path.")


# assign splits where the source had none (deterministic)
for dom in ("primary", "pothole_extra"):
    todo = sorted([it for it in items if it["domain"] == dom and it["split"] is None], key=lambda it: it["img"])
    random.Random(SEED).shuffle(todo)
    n = len(todo); a = int(n * SPLIT[0]); b = int(n * (SPLIT[0] + SPLIT[1]))
    for i, it in enumerate(todo):
        it["split"] = "train" if i < a else ("val" if i < b else "test")

# a primary dataset with only train/val (no test split): move part of val to test so there is a held-out set
prim = [it for it in items if it["domain"] == "primary"]
if prim and not any(it["split"] == "test" for it in prim):
    val = sorted([it for it in prim if it["split"] == "val"], key=lambda it: it["img"])
    random.Random(SEED).shuffle(val)
    for it in val[: len(val) // 2]:
        it["split"] = "test"
    print(f"No test split in your India data: half of val ({len(val) // 2} images) is used as the test split.")

# cap extra-country images, keeping the ones with rare classes first
extra = [it for it in items if it["domain"] == "extra_country"]
if len(extra) > MAX_EXTRA_IMAGES:
    random.Random(SEED).shuffle(extra)
    extra.sort(key=lambda it: not any(b[0] in (1, POTHOLE) for b in it["boxes"]))
    drop = {id(it) for it in extra[MAX_EXTRA_IMAGES:]}
    items = [it for it in items if id(it) not in drop]

if not any(it["domain"] == "primary" for it in items):
    explain_inputs()
    raise RuntimeError("No labelled RDD2022 India images found. Read the listing above, fix the input, and rerun "
                       "from cell 2.")

summary = Counter((it["domain"], it["split"]) for it in items)
for k in sorted(summary):
    print(f"{k[0]:15s} {k[1]:6s} {summary[k]:6d}")

# %%
# 5. Leakage check: near-duplicate photos shared between train and val/test are moved to train
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path, n=16):
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
        return np.packbits((a[:, 1:] > a[:, :-1]).flatten())
    except Exception:
        return None


t0 = time.time()
for it in items:
    it["hash"] = dhash(it["img"])
train_list = [it["hash"] for it in items if it["split"] == "train" and it["hash"] is not None]
train_h = np.stack(train_list) if train_list else np.zeros((0, 32), np.uint8)
flagged = []
for split in (("val", "test") if len(train_h) else ()):
    group = [it for it in items if it["split"] == split and it["domain"] == "primary" and it["hash"] is not None
             and 16 <= POP[it["hash"]].sum() <= 240]  # skip near-blank images, their hashes are meaningless
    for i in range(0, len(group), 64):
        chunk = group[i:i + 64]
        H = np.stack([it["hash"] for it in chunk])
        dist = POP[H[:, None, :] ^ train_h[None, :, :]].sum(-1, dtype=np.uint16).min(1)
        flagged += [it for it, dmin in zip(chunk, dist) if dmin <= 8]  # <= 8 of 256 bits differ: same scene
n_eval = sum(1 for it in items if it["split"] in ("val", "test") and it["domain"] == "primary")
if len(flagged) > 0.2 * max(1, n_eval):
    print(f"WARNING: {len(flagged)} of {n_eval} val/test images look like train images, too many to be real "
          "duplicates, so nothing was moved. Inspect a few pairs manually.")
    moved = 0
else:
    for it in flagged:
        it["split"] = "train"
    moved = len(flagged)
print(f"hashing + check: {time.time() - t0:.0f}s, moved {moved} near-duplicate val/test images into train")

# %%
# 6. Write Track A dataset (4 classes) with repeat-factor oversampling of rare classes
def class_table(rows, nc_names):
    import pandas as pd
    df = pd.DataFrame(rows).fillna(0).astype(int).sort_index()
    df.index = [nc_names[i] for i in df.index]
    return df


def build_detection_dataset(root, entries, names, oversample=True, max_bg_frac=BACKGROUND_FRACTION):
    # entries: list of (item, split, boxes)
    if root.exists():
        shutil.rmtree(root)
    stats = defaultdict(Counter)
    by_split = defaultdict(list)
    for it, sp, boxes in entries:
        by_split[sp].append((it, boxes))
    # limit background images in train
    tr = by_split.get("train", [])
    pos = [e for e in tr if e[1]]; bg = [e for e in tr if not e[1]]
    random.Random(SEED).shuffle(bg)
    by_split["train"] = pos + bg[: int(len(pos) * max_bg_frac)]
    rc = {}
    freq = Counter(c for _, b in by_split["train"] for c in {x[0] for x in b})
    if oversample and freq:
        fmax = max(freq.values())
        rc = {c: min(MAX_REPEAT, max(1.0, math.sqrt(fmax / f))) for c, f in freq.items()}
    rng = random.Random(SEED)
    k, name_map = 0, {}
    for sp, lst in by_split.items():
        for it, boxes in lst:
            reps = 1
            if sp == "train" and rc:
                r = max([rc[c] for c in {b[0] for b in boxes}], default=1.0)
                reps = int(r) + (1 if rng.random() < r - int(r) else 0)
            stem, ext = Path(it["img"]).stem, Path(it["img"]).suffix.lower()
            name_map[id(it)] = (sp, f"{k:06d}_{stem}", ext)
            for rep in range(max(1, reps)):
                name = f"{k:06d}_{stem}" + (f"_rep{rep}" if rep else "")
                safe_link(it["img"], root / "images" / sp / (name + ext))
                write_label(root / "labels" / sp / (name + ".txt"), boxes)
                for b in boxes:
                    stats[sp][b[0]] += 1
            k += 1
    yml = dict(path=str(root), train="images/train", val="images/val",
               names={i: n for i, n in enumerate(names)})
    if (root / "images" / "test").exists():
        yml["test"] = "images/test"
    (root / "data.yaml").write_text(yaml.safe_dump(yml, sort_keys=False))
    if rc:
        print("repeat factors:", {names[c]: round(r, 2) for c, r in sorted(rc.items())})
    print("instances per split (after oversampling):")
    print(class_table({sp: dict(c) for sp, c in stats.items()}, names).T.to_string())
    print({sp: len(list((root / "images" / sp).glob("*"))) for sp in by_split}, "images")
    return root / "data.yaml", name_map


ROOT_A = DATA / "rdd4"
entries_a = [(it, it["split"], it["boxes"]) for it in items if it["domain"] in ("primary", "extra_country")]
YAML_A, _ = build_detection_dataset(ROOT_A, entries_a, CLASSES)

# %%
# ## Track A: 4-class road-damage detector
# What changed compared with the YOLO11n baseline (mAP50 0.315): a larger YOLO26 model, 736 px input instead of 640,
# extra RDD2022 countries in training only, repeat-factor sampling for rare classes (transverse cracks), cosine LR,
# longer schedule with early stopping, and a leakage check between splits. Validation and test stay India-only so the
# numbers are comparable with your earlier result.

# %%
# 7. Train Track A
from ultralytics import YOLO

AUG = dict(mosaic=1.0, mixup=0.10, close_mosaic=15, degrees=0.0, translate=0.10, scale=0.5,
           fliplr=0.5, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4)


def train_detector(run_name, data_yaml, epochs, hours):
    model, _ = load_pretrained("det", DET_SIZE)
    model.train(data=str(data_yaml), epochs=epochs, time=hours, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                workers=WORKERS, optimizer="auto", cos_lr=True, patience=PATIENCE, seed=SEED, amp=True,
                cache=False, plots=True, project=str(RUNS), name=run_name, exist_ok=True, **AUG)


if RUN_TRACK_A:
    train_detector("trackA_rdd4", YAML_A, EPOCHS_A, TIME_A)
BEST_A = find_best("trackA_rdd4")
print("Track A weights:", BEST_A)

# %%
# 8. Evaluate Track A on the held-out India test split
import pandas as pd
RESULTS = {}


def evaluate_detector(weights, data_yaml, split, run_name):
    m = YOLO(str(weights))
    met = m.val(data=str(data_yaml), split=split, imgsz=IMGSZ, batch=16, device=EVAL_DEVICE,
                plots=True, project=str(RUNS), name=run_name, exist_ok=True, verbose=False)
    out = dict(precision=float(met.box.mp), recall=float(met.box.mr),
               mAP50=float(met.box.map50), mAP50_95=float(met.box.map), per_class={})
    for i, c in enumerate(met.box.ap_class_index):
        p, r, ap50, ap = met.box.class_result(i)
        out["per_class"][m.names[int(c)]] = dict(precision=float(p), recall=float(r),
                                                 mAP50=float(ap50), mAP50_95=float(ap))
    table = pd.DataFrame(out["per_class"]).T
    table.loc["all"] = [out["precision"], out["recall"], out["mAP50"], out["mAP50_95"]]
    print(table.round(3).to_string())
    return out


if BEST_A:
    RESULTS["trackA_test_india"] = evaluate_detector(BEST_A, YAML_A, "test", "evalA_test")

# %%
# ## Track B: pothole-only detector
# Pothole boxes from every RDD2022 country plus any extra pothole dataset you attached. Reported twice:
# on the **India test split only** (comparable with Track A) and on the combined test split.

# %%
# 9. Build + train Track B
ROOT_B = DATA / "pothole"
entries_b = []
for it in items:
    pb = [(0, *b[1:]) for b in it["boxes"] if b[0] == POTHOLE]
    if it["domain"] == "primary" or it["domain"] == "pothole_extra":
        entries_b.append((it, it["split"], pb))
    elif it["domain"] == "extra_country":
        entries_b.append((it, "train", pb))
YAML_B, NAMES_B = build_detection_dataset(ROOT_B, entries_b, ["pothole"], oversample=False, max_bg_frac=0.15)

# India-only test split for a fair comparison with Track A
for sub in ("images", "labels"):
    shutil.rmtree(ROOT_B / sub / "test_india", ignore_errors=True)
    (ROOT_B / sub / "test_india").mkdir(parents=True)
for it, sp, _ in entries_b:
    if it["domain"] == "primary" and sp == "test" and id(it) in NAMES_B:
        _, name, ext = NAMES_B[id(it)]
        safe_link(it["img"], ROOT_B / "images" / "test_india" / (name + ext))
        shutil.copy(ROOT_B / "labels" / "test" / (name + ".txt"), ROOT_B / "labels" / "test_india" / (name + ".txt"))
d = yaml.safe_load(open(YAML_B)); d["test"] = "images/test_india"
YAML_B_INDIA = ROOT_B / "data_india_test.yaml"
YAML_B_INDIA.write_text(yaml.safe_dump(d, sort_keys=False))
print("India-only pothole test images:", len(list((ROOT_B / "images" / "test_india").glob("*"))))

if RUN_TRACK_B:
    train_detector("trackB_pothole", YAML_B, EPOCHS_B, TIME_B)
BEST_B = Path(TRACK_B_WEIGHTS) if TRACK_B_WEIGHTS else find_best("trackB_pothole")
print("Track B weights:", BEST_B)
if BEST_B:
    print("\nIndia test:"); RESULTS["trackB_test_india"] = evaluate_detector(BEST_B, YAML_B_INDIA, "test", "evalB_india")
    print("\nCombined test:"); RESULTS["trackB_test_all"] = evaluate_detector(BEST_B, YAML_B, "test", "evalB_all")

# %%
# ## Track C: pothole verifier (classifier)
# Every box the detector calls a pothole is cropped and checked by a small classifier. Training crops come from
# ground-truth potholes (positives), random road patches, and **hard negatives**: boxes the Track B detector wrongly
# called potholes. Crops keep the split of the photo they came from, so no test photo leaks into training.

# %%
# 10. Build the verifier dataset
ROOT_C = DATA / "verifier"
if ROOT_C.exists():
    shutil.rmtree(ROOT_C)


def crop(im, cx, cy, w, h, pad=0.20, min_px=32):
    W, H = im.size
    bw, bh = max(w * W * (1 + 2 * pad), min_px), max(h * H * (1 + 2 * pad), min_px)
    x0, y0 = max(0, cx * W - bw / 2), max(0, cy * H - bh / 2)
    x1, y1 = min(W, cx * W + bw / 2), min(H, cy * H + bh / 2)
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


def iou_xywh(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


# hard negatives from the Track B detector (if it exists)
hard = defaultdict(list)
src_items = [(it, sp, [(0, *b[1:]) for b in it["boxes"] if b[0] == POTHOLE]) for it, sp, _ in entries_b]
if BEST_B:
    det_b = YOLO(str(BEST_B))
    paths = [it["img"] for it, _, _ in src_items]
    for i in range(0, len(paths), 64):
        for r in det_b.predict(paths[i:i + 64], conf=0.15, imgsz=IMGSZ, device=EVAL_DEVICE, verbose=False):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for xywhn in r.boxes.xywhn.cpu().numpy():
                hard[r.path].append(tuple(map(float, xywhn)))

rng = random.Random(SEED)
pos_sizes = [(b[3], b[4]) for _, _, bs in src_items for b in bs] or [(0.1, 0.1)]

# pass 1: plan crops so that negatives = 1.5 x positives per split (hard negatives first, capped at 1 x positives)
plan = [dict(hard=[], rand=0) for _ in src_items]
for sp in ("train", "val", "test"):
    idxs = [i for i, (_, s_, _) in enumerate(src_items) if s_ == sp]
    n_pos = sum(len(src_items[i][2]) for i in idxs)
    cand = []
    for i in idxs:
        it, _, gts = src_items[i]
        for hb in hard.get(it["img"], [])[:3]:
            if all(iou_xywh(hb, g[1:]) < 0.10 for g in gts):
                cand.append((i, hb))
    rng.shuffle(cand)
    for i, hb in cand[:n_pos]:
        plan[i]["hard"].append(hb)
    n_rand = max(0, int(1.5 * n_pos) - min(len(cand), n_pos))
    for i in (rng.choices(idxs, k=n_rand) if idxs else []):
        plan[i]["rand"] += 1

# pass 2: write crops
counts = Counter()
for i, (it, sp, gts) in enumerate(src_items):
    if not gts and not plan[i]["hard"] and not plan[i]["rand"]:
        continue
    try:
        im = Image.open(it["img"]).convert("RGB")
    except Exception:
        continue
    stem = f"{i:06d}_{Path(it['img']).stem}"
    out_p, out_n = ROOT_C / sp / "pothole", ROOT_C / sp / "not_pothole"
    out_p.mkdir(parents=True, exist_ok=True); out_n.mkdir(parents=True, exist_ok=True)
    for j, (_, cx, cy, w, h) in enumerate(gts):
        crop(im, cx, cy, w, h).save(out_p / f"{stem}_p{j}.jpg", quality=92); counts[(sp, "pos")] += 1
    for j, hb in enumerate(plan[i]["hard"]):
        crop(im, *hb).save(out_n / f"{stem}_h{j}.jpg", quality=92); counts[(sp, "hard_neg")] += 1
    made, tries = 0, 0
    while made < plan[i]["rand"] and tries < 30:
        tries += 1
        w, h = rng.choice(pos_sizes)
        cx, cy = rng.uniform(w / 2, 1 - w / 2), rng.uniform(max(h / 2, 0.35), max(0.36, 1 - h / 2))
        if all(iou_xywh((cx, cy, w, h), g[1:]) < 0.05 for g in gts):
            crop(im, cx, cy, w, h).save(out_n / f"{stem}_r{made}.jpg", quality=92); counts[(sp, "rand_neg")] += 1
            made += 1
    im.close()
for sp in ("train", "val", "test"):
    print(sp, {k: counts[(sp, k)] for k in ("pos", "hard_neg", "rand_neg")})

# %%
# 11. Train + evaluate the verifier
if RUN_TRACK_C:
    model_c, _ = load_pretrained("cls", CLS_SIZE)
    model_c.train(data=str(ROOT_C), epochs=EPOCHS_C, time=TIME_C, imgsz=224, batch=64, device=EVAL_DEVICE,
                  workers=WORKERS, patience=15, seed=SEED, amp=True, fliplr=0.5, hsv_v=0.4, erasing=0.2,
                  project=str(RUNS), name="trackC_verifier", exist_ok=True, plots=True)
BEST_C = find_best("trackC_verifier")
print("Track C weights:", BEST_C)


def pothole_index(cls_model):
    return [k for k, v in cls_model.names.items() if v == "pothole"][0]


def verifier_scores(ver, images):
    pi, scores = pothole_index(ver), []
    for i in range(0, len(images), 128):
        batch = [str(x) if isinstance(x, Path) else x for x in images[i:i + 128]]
        for r in ver.predict(batch, imgsz=224, device=EVAL_DEVICE, verbose=False):
            scores.append(float(r.probs.data[pi]))
    return np.array(scores)


if BEST_C:
    ver = YOLO(str(BEST_C))
    pos = sorted((ROOT_C / "test" / "pothole").glob("*.jpg")); neg = sorted((ROOT_C / "test" / "not_pothole").glob("*.jpg"))
    s = verifier_scores(ver, pos + neg); y = np.array([1] * len(pos) + [0] * len(neg)); yhat = s >= 0.5
    tp, fp = int((yhat & (y == 1)).sum()), int((yhat & (y == 0)).sum())
    fn, tn = int((~yhat & (y == 1)).sum()), int((~yhat & (y == 0)).sum())
    RESULTS["trackC_test"] = dict(accuracy=(tp + tn) / max(1, len(y)), precision=tp / max(1, tp + fp),
                                  recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
                                  confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn), n=len(y))
    hard_mask = np.array([False] * len(pos) + ["_h" in f.stem for f in neg])
    if hard_mask.any():  # the hardest case: boxes the detector got wrong
        RESULTS["trackC_test"]["hard_negative_rejection"] = float((~yhat[hard_mask]).mean())
    print(json.dumps(RESULTS["trackC_test"], indent=2))

# %%
# ## End-to-end evaluation on India test photos
# Thresholds are chosen on the **validation** split and then applied once to the test split.
# * *Box level*: a predicted pothole counts as correct if IoU ≥ 0.5 with a real pothole.
# * *Image level*: the photo is correctly classified as "has a pothole" / "no pothole". This is the decision the web
#   app makes before it files a report.

# %%
# 12. Detector-only vs detector + verifier
def box_iou(a, b):  # a: (4,), b: (n,4) xyxy
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def collect(det, ver, img_dir, lbl_dir):
    files = sorted(img_dir.glob("*")); preds, gts = {}, {}
    for i in range(0, len(files), 32):
        for r in det.predict([str(f) for f in files[i:i + 32]], conf=0.05, imgsz=IMGSZ, device=EVAL_DEVICE, verbose=False):
            key = Path(r.path).stem; H, W = r.orig_shape
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            v = np.ones(len(b))
            if ver is not None and len(b):
                pil = Image.fromarray(np.ascontiguousarray(r.orig_img[..., ::-1]))  # BGR -> RGB
                crops = [crop(pil, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
                         for x0, y0, x1, y1 in b]
                v = verifier_scores(ver, crops)
            preds[key] = (b, c, v)
            g = []
            lf = lbl_dir / f"{key}.txt"
            if lf.exists():
                for line in lf.read_text().split("\n"):
                    if line.strip():
                        _, cx, cy, w, h = map(float, line.split())
                        g.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            gts[key] = np.array(g).reshape(-1, 4)
    return preds, gts


def score(preds, gts, ct, vt=None, iou_t=0.5):
    tp = fp = fn = 0; img_ok = 0
    for k, g in gts.items():
        b, c, v = preds.get(k, (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
        keep = c >= ct
        if vt is not None:
            keep &= v >= vt
        b, c = b[keep], c[keep]
        b = b[np.argsort(-c)]
        used = np.zeros(len(g), bool)
        for box in b:
            if len(g):
                ious = box_iou(box, g); ious[used] = 0; j = int(ious.argmax())
                if ious[j] >= iou_t:
                    used[j] = True; tp += 1; continue
            fp += 1
        fn += int((~used).sum())
        img_ok += int((len(b) > 0) == (len(g) > 0))
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(precision=p, recall=r, f1=2 * p * r / max(1e-9, p + r), image_accuracy=img_ok / max(1, len(gts)),
                conf=ct, verifier=vt, n_images=len(gts))


DEPLOY = None
if BEST_B:
    det = YOLO(str(BEST_B)); ver = YOLO(str(BEST_C)) if BEST_C else None
    val_p, val_g = collect(det, ver, ROOT_B / "images" / "val", ROOT_B / "labels" / "val")
    test_p, test_g = collect(det, ver, ROOT_B / "images" / "test_india", ROOT_B / "labels" / "test_india")
    grid = np.round(np.arange(0.10, 0.70, 0.05), 2)
    best_det = max((score(val_p, val_g, ct) for ct in grid), key=lambda d: d["f1"])
    rows = {"detector only": score(test_p, test_g, best_det["conf"])}
    best = best_det
    if ver is not None:
        best = max((score(val_p, val_g, ct, vt) for ct in grid for vt in (0.3, 0.4, 0.5, 0.6, 0.7)),
                   key=lambda d: d["f1"])
        rows["detector + verifier"] = score(test_p, test_g, best["conf"], best["verifier"])
    RESULTS["end_to_end_test_india"] = rows
    print(pd.DataFrame(rows).T.round(3).to_string())
    DEPLOY = dict(det_conf=float(best["conf"]),
                  verifier_threshold=float(best["verifier"]) if best.get("verifier") is not None else None,
                  imgsz=IMGSZ, weights=USED_WEIGHTS)
    print("deploy thresholds:", DEPLOY)

# %%
# 13. Look at a few test predictions
import matplotlib.pyplot as plt
w = BEST_B or BEST_A
if w:
    m = YOLO(str(w))
    sample = sorted((ROOT_B / "images" / "test_india").glob("*"))[:8] or sorted((ROOT_A / "images" / "test").glob("*"))[:8]
    if sample:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for ax in axes.flat:
            ax.axis("off")
        for ax, f in zip(axes.flat, sample):
            r = m.predict(str(f), conf=DEPLOY["det_conf"] if DEPLOY else 0.3, imgsz=IMGSZ,
                          device=EVAL_DEVICE, verbose=False)[0]
            ax.imshow(r.plot()[..., ::-1]); ax.set_title(f"{len(r.boxes)} detection(s)")
        plt.tight_layout(); plt.savefig(EXPORT / "sample_predictions.jpg", dpi=80)

# %%
# 14. Export everything the web app needs
import zipfile
for src, dst in [(BEST_A, "road_damage_4class.pt"), (BEST_B, "pothole_detector.pt"), (BEST_C, "pothole_verifier.pt")]:
    if src and Path(src).exists():
        shutil.copy(src, EXPORT / dst)
if DEPLOY:
    (EXPORT / "deploy_config.json").write_text(json.dumps(DEPLOY, indent=2))
(EXPORT / "metrics_summary.json").write_text(json.dumps(RESULTS, indent=2, default=float))
plots = EXPORT / "plots"; plots.mkdir(exist_ok=True)
for run in RUNS.iterdir():
    for f in list(run.glob("*.png")) + list(run.glob("*.jpg")) + list(run.glob("results.csv")):
        shutil.copy(f, plots / f"{run.name}__{f.name}")
with zipfile.ZipFile(WORK / "pothole_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in EXPORT.rglob("*"):
        z.write(f, f.relative_to(EXPORT))
print("Saved:", sorted(p.name for p in EXPORT.iterdir()))
print("Download /kaggle/working/pothole_models.zip from the Output tab and unzip it into webapp/models/")

# %%
# 15. Numbers for your slides (copy the ones you actually got)
def pct(x):
    return f"{100 * x:.1f}%"

lines = []
det_name = USED_WEIGHTS.get("det", f"{MODEL_FAMILY}{DET_SIZE}.pt").replace(".pt", "")
if "trackA_test_india" in RESULTS:
    a = RESULTS["trackA_test_india"]
    lines.append(f"4-class detector ({det_name}, India test): mAP50 {a['mAP50']:.3f}, "
                 f"mAP50-95 {a['mAP50_95']:.3f}, P {a['precision']:.3f}, R {a['recall']:.3f}")
if "trackB_test_india" in RESULTS:
    b = RESULTS["trackB_test_india"]
    lines.append(f"Pothole detector (India test): mAP50 {b['mAP50']:.3f}, P {b['precision']:.3f}, R {b['recall']:.3f}")
if "trackC_test" in RESULTS:
    c = RESULTS["trackC_test"]
    lines.append(f"Pothole verifier (test crops, n={c['n']}): accuracy {pct(c['accuracy'])}, "
                 f"precision {pct(c['precision'])}, recall {pct(c['recall'])}")
for name, r in RESULTS.get("end_to_end_test_india", {}).items():
    lines.append(f"End-to-end {name} (n={r['n_images']} India test photos): box P {pct(r['precision'])}, "
                 f"R {pct(r['recall'])}, F1 {pct(r['f1'])}, image-level accuracy {pct(r['image_accuracy'])}")
print("\n".join(lines) or "No results yet.")
(EXPORT / "slide_numbers.txt").write_text("\n".join(lines))
with zipfile.ZipFile(WORK / "pothole_models.zip", "a") as z:  # add it to the download too
    z.write(EXPORT / "slide_numbers.txt", "slide_numbers.txt")

# %%
# 16. Tidy /kaggle/working so the saved version only holds results (the built datasets are symlinks + crops
#     that can be rebuilt in a minute; leaving them makes Kaggle's output upload slow)
CLEAN_UP_DATA = True
if CLEAN_UP_DATA:
    shutil.rmtree(DATA, ignore_errors=True)
print("Kept:", sorted(p.name for p in WORK.iterdir()))
