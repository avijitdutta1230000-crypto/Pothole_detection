# PotholeWatch: AI pothole detection with geotagged municipal reporting

BCSE497J Project-I. Two parts:

| Folder | What it is |
|---|---|
| `kaggle/` | Training pipeline for Kaggle: YOLO26 4-class road-damage detector, pothole-only detector, pothole verifier, honest evaluation, export for the web app |
| `webapp/` | Flask + SQLite web app: citizen reporting, authority dashboard, email/SMS notifications both ways, drive mode with 1 km GPS warnings |

---

## Part 1: Train the models on Kaggle

1. Create a new Kaggle notebook and choose *File → Import Notebook*, then upload `kaggle/pothole_yolo26_kaggle.ipynb`.
   (`pothole_yolo26_kaggle.py` is the same code as a plain script, if you prefer a Script kernel.)
2. *Settings → Accelerator → GPU T4 x2*, and *Settings → Internet → On*.
3. *Add Input* → your RDD2022 dataset (your existing 4-class YOLO version, or raw RDD2022 with VOC XML).
   Optionally add any pothole dataset too; it is used only for the pothole detector.
4. *Save Version → Save & Run All (Commit)*. With the default settings the run takes about 9.5 hours and
   finishes inside one 12-hour session. For a quick check first, set `TIME_A = TIME_B = TIME_C = None` and
   `EPOCHS_A = EPOCHS_B = EPOCHS_C = 3` in cell 2.
5. From the *Output* tab download `pothole_models.zip`. It contains:
   `pothole_detector.pt`, `pothole_verifier.pt`, `road_damage_4class.pt`, `deploy_config.json` (thresholds tuned on
   the validation split), `metrics_summary.json`, `slide_numbers.txt` and all training/evaluation plots.

**YOLO26 vs YOLO27.** YOLO26 is the newest Ultralytics model with published weights. YOLO27 has been announced
but not released. Cell 2 sets `MODEL_FAMILY = "yolo27"`; until those weights exist the notebook falls back
to YOLO26 automatically (you'll see `yolo27s.pt unavailable … Pretrained weights: yolo26s.pt` in the log).

**What changed from the YOLO11n baseline (mAP50 0.315):** YOLO26s instead of 11n, 736 px input instead of 640 (India
photos are 720×720), other RDD2022 countries added to training only, repeat-factor oversampling for rare classes,
a near-duplicate leakage check between train and val/test, cosine LR with early stopping, and a second-stage
verifier that removes false pothole boxes.

**About 92–95 %.** Published, architecture-modified YOLOv8 models reach about 62–63 % mAP50 on RDD2022, so
don't expect the 4-class mAP50 in the 90s. The numbers that can realistically be that high are the verifier's
crop accuracy and the end-to-end image-level accuracy ("does this photo contain a pothole?"). All metrics are
measured on the held-out India test split. Report each number with its metric name; `slide_numbers.txt`
already words them that way.

---

## Part 2: Run the web app

Needs Python 3.10+.

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env   (then edit it)
# unzip pothole_models.zip from Kaggle into webapp/models/
python app.py
```

Open http://localhost:5000. The authority side is at http://localhost:5000/admin (email and password come from
`ADMIN_EMAIL` / `ADMIN_PASSWORD` in `.env` the first time the app starts; change the password later under *Account*).

**Demo data:** `python seed_demo.py` adds two authority contacts and seven reports around VIT Vellore, so the
dashboard and drive mode have something to show even before your models are ready. To start over, delete
`webapp/instance/potholes.db` and `webapp/uploads/`.

### Notifications

| When | Who is notified | How |
|---|---|---|
| New pothole reported | The authority contact whose zone contains it (or the city-wide contact) | Email with the annotated photo + SMS with a map link |
| Same pothole reported again (within 15 m) | Authority again if severity went up, or at the 3rd/5th/10th report | Email + SMS |
| Report received | The citizen, if they left an email or phone | Email + SMS with a tracking link |
| Repair started / fixed / closed | Every citizen who reported that pothole | Email (with the after-repair photo) + SMS; plus a browser pop-up if their tracking page is open |

Every message is logged on the admin *Notifications* page. Without credentials in `.env` messages are logged as
*Not set up* instead of being sent, so the whole flow still works in a demo.

**Email (Gmail):** turn on 2-Step Verification for the Gmail account, create an App Password
(Google Account → Security → 2-Step Verification → App passwords), then in `.env`:
`SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`, `SMTP_USER=you@gmail.com`, `SMTP_PASSWORD=<the 16-letter app password>`.

**SMS (Twilio):** create a Twilio account, copy the Account SID, Auth Token and your Twilio phone number into
`.env`. A trial account can only send to numbers you verify in the Twilio console, which is enough for a demo.

Use *Contacts → Send test* to check both channels.

### Drive mode (GPS warning within 1 km)

`/drive` watches the phone's GPS and checks open potholes around it. It warns three times as you approach: at
1 km, at 300 m and at 80 m (voice, vibration, a large on-screen sign; a system notification if the tab is in the
background). Potholes behind you are ignored once your direction of travel is known. Fixed potholes stop
triggering warnings. Change the radius with `ALERT_RADIUS_M` in `.env`.
Tick *Simulate* and tap the map to demo it from a laptop.

### Using it on a phone

Phones only share GPS with **https** pages. Two options:

* `python app.py --https` serves a self-signed certificate (needs `pyopenssl`, already in requirements). Open
  `https://<your-laptop-ip>:5000` on the phone (same Wi-Fi) and accept the browser warning once.
* Or run a tunnel, for example `cloudflared tunnel --url http://localhost:5000`, and open the https address it
  prints. Put that address in `PUBLIC_BASE_URL` so links in emails and SMS work.

### How the app decides things

* **Severity** (slide 17): 50 % damage type + 35 % box size + 15 % detection confidence →
  LOW < 0.42 ≤ MEDIUM < 0.72 ≤ HIGH. Weights are at the top of `webapp/severity.py`.
* **Two-stage check:** every pothole box from the detector is cropped and checked by the verifier; boxes below the
  tuned threshold are dropped before a report is filed.
* **Location:** browser GPS, or a pin the user drags, or GPS stored in the photo's EXIF as a fallback; reverse
  geocoded with OpenStreetMap Nominatim (max 1 request per second, cached).
* **Routing:** each authority contact can have a zone (centre + radius); the smallest zone containing the pothole
  wins, otherwise a contact without a zone gets it.

### Tests

`python tests/test_flows.py` (from `webapp/`) runs 39 end-to-end checks: report → authority notified, duplicate
merge, zone routing, EXIF fallback, 1 km hazard search, status changes → citizen notified, tracking page,
accounts.

### Known limits

* Browser pop-ups only work while the tracking page is open; email/SMS are the reliable channel.
* Drive mode needs the screen on (it asks the phone to keep it awake).
* `python app.py` is Flask's development server and SQLite is a single file: fine for a demo or a pilot,
  not for a city-wide deployment.

## Project structure

```
kaggle/
  pothole_yolo26_kaggle.ipynb   training notebook (run on Kaggle)
  pothole_yolo26_kaggle.py      same code as a script
  build_notebook.py             generates both from one source
webapp/
  app.py            routes: citizen pages, API, admin
  detector.py       YOLO26 detector + verifier, annotated photo
  severity.py       severity score
  notifier.py       email (SMTP) + SMS (Twilio), message text
  geo.py            distance, bearing, reverse geocoding, EXIF GPS
  db.py             SQLite schema
  config.py         settings from .env
  seed_demo.py      demo data around VIT Vellore
  templates/, static/   pages, CSS, JavaScript (Leaflet maps)
  models/           put the .pt files from Kaggle here
  tests/            end-to-end flow test
```
