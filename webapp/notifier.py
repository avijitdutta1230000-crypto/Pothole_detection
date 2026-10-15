"""Sends notifications by email (SMTP) and SMS (Twilio) in a background thread.

Every message is written to the `notifications` table. If a channel has no credentials in .env the message is
stored with status `not_configured`, so the full flow can still be demonstrated from the admin
Notifications page.
"""
import mimetypes
import re
import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from pathlib import Path

import requests

from db import connect, now_iso

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def valid_email(value):
    return bool(value and EMAIL_RE.match(value.strip()))


def normalize_phone(value, default_cc="+91"):
    """'98765 43210' -> '+919876543210'. Returns None when it cannot be a phone number."""
    if not value:
        return None
    digits = re.sub(r"[^\d+]", "", value)
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    if not digits.startswith("+"):
        if len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        digits = (default_cc + digits) if len(digits) == 10 else "+" + digits
    return digits if 9 <= len(digits) - 1 <= 15 and digits[1:].isdigit() else None


class Notifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db_path = cfg["DATABASE"]
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notify")

    @property
    def email_ready(self):
        c = self.cfg
        return bool(c["SMTP_HOST"] and c["SMTP_USER"] and c["SMTP_PASSWORD"])

    @property
    def sms_ready(self):
        c = self.cfg
        return bool(c["TWILIO_ACCOUNT_SID"] and c["TWILIO_AUTH_TOKEN"] and c["TWILIO_FROM_NUMBER"])

    # ---------------------------------------------------------------- transport
    def _send_email(self, to, subject, text, attachment=None):
        c = self.cfg
        msg = EmailMessage()
        msg["From"] = f"PotholeWatch <{c['SMTP_FROM']}>"
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(text)
        if attachment and Path(attachment).exists():
            ctype = mimetypes.guess_type(str(attachment))[0] or "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            msg.add_attachment(Path(attachment).read_bytes(), maintype=maintype, subtype=subtype,
                               filename=Path(attachment).name)
        ctx = ssl.create_default_context()
        if c["SMTP_USE_SSL"]:
            with smtplib.SMTP_SSL(c["SMTP_HOST"], c["SMTP_PORT"], context=ctx, timeout=20) as s:
                s.login(c["SMTP_USER"], c["SMTP_PASSWORD"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(c["SMTP_HOST"], c["SMTP_PORT"], timeout=20) as s:
                s.starttls(context=ctx)
                s.login(c["SMTP_USER"], c["SMTP_PASSWORD"])
                s.send_message(msg)

    def _send_sms(self, to, body):
        c = self.cfg
        url = f"https://api.twilio.com/2010-04-01/Accounts/{c['TWILIO_ACCOUNT_SID']}/Messages.json"
        r = requests.post(url, data={"From": c["TWILIO_FROM_NUMBER"], "To": to, "Body": body},
                          auth=(c["TWILIO_ACCOUNT_SID"], c["TWILIO_AUTH_TOKEN"]), timeout=20)
        if r.status_code >= 300:
            try:
                detail = r.json().get("message", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"Twilio {r.status_code}: {detail[:200]}")

    def _log(self, report_id, audience, channel, recipient, subject, body, status, error=None):
        con = connect(self.db_path)
        try:
            con.execute(
                "INSERT INTO notifications (report_id, audience, channel, recipient, subject, body, status, error,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (report_id, audience, channel, recipient, subject, body, status, error, now_iso()),
            )
            con.commit()
        except Exception as e:  # a failed log write must not leave the database locked
            con.rollback()
            print(f"[notify] could not record {channel} to {recipient}: {e}")
        finally:
            con.close()

    def _deliver(self, report_id, audience, channel, recipient, subject, body, attachment):
        ready = self.email_ready if channel == "email" else self.sms_ready
        if not ready:
            print(f"[notify:{channel} not configured] to {recipient}: {subject or body[:80]}")
            return self._log(report_id, audience, channel, recipient, subject, body, "not_configured")
        try:
            if channel == "email":
                self._send_email(recipient, subject, body, attachment)
            else:
                self._send_sms(recipient, body)
            self._log(report_id, audience, channel, recipient, subject, body, "sent")
        except Exception as e:  # never let a notification failure break a request
            self._log(report_id, audience, channel, recipient, subject, body, "failed", str(e)[:500])

    # ---------------------------------------------------------------- public
    def send(self, report_id, audience, recipients, subject, email_text, sms_text, attachment=None, wait=False):
        """recipients: iterable of dicts with optional 'email' / 'phone' keys (and 'notify_email'/'notify_sms')."""
        futures = []
        seen = set()
        for r in recipients:
            email, phone = (r.get("email") or "").strip(), (r.get("phone") or "").strip()
            if email and r.get("notify_email", 1) and ("email", email) not in seen:
                seen.add(("email", email))
                futures.append(self.pool.submit(self._deliver, report_id, audience, "email", email,
                                                subject, email_text, attachment))
            if phone and r.get("notify_sms", 1) and ("sms", phone) not in seen:
                seen.add(("sms", phone))
                futures.append(self.pool.submit(self._deliver, report_id, audience, "sms", phone,
                                                None, sms_text, None))
        if wait:
            for f in futures:
                f.result()
        return len(futures)


# -------------------------------------------------------------------- message text
def maps_link(lat, lng):
    return f"https://www.google.com/maps?q={lat:.6f},{lng:.6f}"


def msg_new_report(r, base_url, repeat=False):
    what = f"{r['severity'].title()}-severity {r['primary_label'].lower()}"
    head = f"Reported again ({r['report_count']} reports so far)" if repeat else "New road damage report"
    subject = f"[PotholeWatch] {what} at {r['address']}"
    email = (
        f"{head}\n\n"
        f"Report code: {r['code']}\n"
        f"Damage: {what} (score {r['severity_score']:.2f}, confidence {(r['confidence'] or 0):.2f})\n"
        f"Location: {r['address']}\n"
        f"Map: {maps_link(r['lat'], r['lng'])}\n"
        f"Reported: {r['created_at']} UTC\n\n"
        f"Open it in the dashboard to start work or mark it fixed:\n{base_url}/admin?report={r['code']}\n\n"
        f"The annotated photo is attached."
    )
    sms = (f"PotholeWatch: {what} {'reported again' if repeat else 'reported'} at {r['address'][:60]}. "
           f"Map {maps_link(r['lat'], r['lng'])} Code {r['code']}")
    return subject, email, sms


def msg_citizen_received(r, base_url):
    subject = f"We received your report {r['code']}"
    url = f"{base_url}/track/{r['code']}"
    email = (f"Thank you. Your report of a {r['primary_label'].lower()} at {r['address']} has been sent to the "
             f"municipal road team.\n\nTrack it here: {url}\n\nWe will message you when it is fixed.")
    sms = f"PotholeWatch: report {r['code']} received and sent to the road team. Track: {url}"
    return subject, email, sms


def msg_citizen_status(r, status, note, authority_name, base_url):
    url = f"{base_url}/track/{r['code']}"
    who = authority_name or "The municipal road team"
    if status == "IN_PROGRESS":
        subject = f"Repair started on {r['code']}"
        body = f"{who} has started work on the {r['primary_label'].lower()} you reported at {r['address']}."
        sms = f"PotholeWatch: repair started on {r['code']} ({r['address'][:50]}). {url}"
    elif status == "RESOLVED":
        subject = f"Fixed: the pothole you reported ({r['code']})"
        body = f"{who} marked the {r['primary_label'].lower()} at {r['address']} as fixed."
        sms = f"PotholeWatch: {r['code']} at {r['address'][:50]} is fixed. Thank you for reporting. {url}"
    else:  # REJECTED
        subject = f"Update on your report {r['code']}"
        body = f"{who} closed your report of a {r['primary_label'].lower()} at {r['address']} without repair."
        sms = f"PotholeWatch: {r['code']} was closed by the road team. Details: {url}"
    if note:
        body += f"\n\nNote from the road team: {note}"
    body += f"\n\nSee the photos and history: {url}"
    return subject, body, sms


def msg_reopened(r, base_url, passes):
    """Passing pothole boxes still find a pothole that was marked fixed."""
    url = f"{base_url}/admin?report={r['code']}"
    subject = f"[PotholeWatch] Reopened: pothole still there at {r['address']}"
    email = (
        f"Report {r['code']} was marked fixed, but {passes} separate passes by pothole boxes still found a pothole "
        f"at this spot, so the complaint has been reopened.\n\n"
        f"Location: {r['address']}\nMap: {maps_link(r['lat'], r['lng'])}\n\n"
        f"Please check the repair:\n{url}\n\nThe latest photo from the passing vehicle is attached."
    )
    sms = (f"PotholeWatch: {r['code']} at {r['address'][:50]} was marked fixed but vehicles still find it. "
           f"Reopened. {maps_link(r['lat'], r['lng'])}")
    return subject, email, sms


def msg_citizen_reopened(r, base_url):
    url = f"{base_url}/track/{r['code']}"
    subject = f"Update: the pothole you reported ({r['code']}) is still there"
    body = (f"The {r['primary_label'].lower()} at {r['address']} was marked fixed, but vehicles passing that spot "
            f"still find it. We have reopened the complaint and told the road team again.\n\nTrack it here: {url}")
    sms = f"PotholeWatch: {r['code']} is still there after the repair, so it has been reopened. {url}"
    return subject, body, sms
