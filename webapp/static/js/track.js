(() => {
  const root = document.getElementById("track");
  if (!root) return;
  const code = root.dataset.code;
  let status = root.dataset.status;
  const FINAL = ["RESOLVED", "REJECTED"];
  const btn = document.getElementById("notify-btn");
  const hint = document.getElementById("notify-hint");

  function refreshHint() {
    if (!btn) return;
    if (!("Notification" in window)) {
      btn.hidden = true;
      hint.textContent = "This browser can't show notifications. We'll still email or text you if you left your details.";
    } else if (Notification.permission === "granted") {
      btn.hidden = true;
      hint.textContent = "Notifications are on. Keep this page open in a tab and you'll get a pop-up when the status changes.";
    } else if (Notification.permission === "denied") {
      btn.hidden = true;
      hint.textContent = "Notifications are blocked for this site in your browser settings.";
    }
  }
  btn?.addEventListener("click", async () => { await Notification.requestPermission(); refreshHint(); });
  refreshHint();

  const messages = {
    IN_PROGRESS: "Repair has started on the pothole you reported.",
    RESOLVED: "The pothole you reported has been fixed.",
    REJECTED: "Your report was closed by the road team.",
    OPEN: "Your report was reopened.",
  };

  async function poll() {
    if (FINAL.includes(status)) return;
    const data = await PW.api(`/api/reports/${encodeURIComponent(code)}`);
    if (data.ok && data.report.status !== status) {
      status = data.report.status;
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification(`PotholeWatch ${code}`, { body: messages[status] || "Status updated.", tag: code });
      }
      location.reload();
      return;
    }
    setTimeout(poll, 20000);
  }
  setTimeout(poll, 20000);
})();
