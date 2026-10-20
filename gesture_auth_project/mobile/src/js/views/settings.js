// WaveLock mobile: Settings (replaces web/js/views/settings.js, whose camera
// scan and server options do not exist on a phone).

import { api } from "../api.js";
import { h, icon, toast, notice, add } from "../ui.js";

function toggle(label, on, onChange) {
  return h("button", { class: "switch-row", type: "button", role: "switch", "aria-checked": String(on),
    onclick: (e) => onChange(!on, e.currentTarget) },
  h("span", null, label), h("span", { class: `pill sm ${on ? "ok" : ""}` }, on ? "On" : "Off"));
}

export default {
  id: "settings",
  navLabel: "Settings",
  title: "Settings",
  subtitle: "Appearance, camera, mode and storage.",
  icon: "settings",
  camera: false,

  async mount(root, app) {
    const sys = app.system || {};
    const mob = sys.mobile || {};

    const pref = app.themePreference();
    const seg = h("div", { class: "segmented", role: "group", "aria-label": "Theme" });
    for (const [value, ic, label] of [["light", "sun", "Light"], ["dark", "moon", "Dark"], ["system", "monitor", "System"]]) {
      seg.append(h("button", { type: "button", "aria-pressed": String(pref === value), onclick: (e) => {
        app.setTheme(value);
        for (const b of seg.children) b.setAttribute("aria-pressed", "false");
        e.currentTarget.setAttribute("aria-pressed", "true");
      } }, icon(ic), label));
    }

    const camSeg = h("div", { class: "segmented", role: "group", "aria-label": "Camera" });
    for (const [value, label] of [["user", "Front (selfie)"], ["environment", "Back"]]) {
      camSeg.append(h("button", { type: "button", "aria-pressed": String(mob.facing === value), onclick: async (e) => {
        const btn = e.currentTarget;            // null after the first await
        try {
          await api.setOptions({ facing: value });
          await api.cameraStop();
          for (const b of camSeg.children) b.setAttribute("aria-pressed", "false");
          btn.setAttribute("aria-pressed", "true");
          toast(value === "user" ? "The front camera will be used." : "The back camera will be used.", "ok");
          app.poll();
        } catch (ex) { toast(ex.message, "warn"); }
      } }, icon("camera"), label));
    }

    const reload = (text) => { toast(text, "ok"); setTimeout(() => location.reload(), 600); };
    const storageText = h("p", { class: "small muted" }, "Checking storage…");
    const persistBtn = h("button", { class: "btn sm", type: "button", onclick: async () => {
      const ok = navigator.storage && navigator.storage.persist ? await navigator.storage.persist() : false;
      toast(ok ? "The browser will keep the profiles even when storage runs low." : "The browser did not grant persistent storage.", ok ? "ok" : "warn");
      paintStorage();
    } }, icon("lock"), "Keep profiles");
    const paintStorage = async () => {
      try {
        const persisted = navigator.storage && navigator.storage.persisted ? await navigator.storage.persisted() : false;
        const est = navigator.storage && navigator.storage.estimate ? await navigator.storage.estimate() : null;
        storageText.textContent = (est ? `This site uses about ${Math.round(est.usage / 1e6)} MB on this device (app files and profiles). ` : "") +
          (persisted ? "Storage is persistent: the browser will not clear it on its own." :
            "Storage is not persistent: the browser may clear it when the device runs low on space.");
        persistBtn.hidden = persisted;
      } catch (e) { storageText.textContent = ""; }
    };
    paintStorage();

    add(root,
      h("div", { class: "grid grid-2" },
        h("div", { class: "card stack" },
          h("h2", null, "Appearance"),
          h("p", { class: "small muted" }, "Light, dark, or follow the phone. Saved on this device."),
          seg),
        h("div", { class: "card stack" },
          h("h2", null, "Camera"),
          h("p", { class: "small muted" }, "Hold the phone the same way (upright or sideways) as during registration."),
          camSeg)),
      h("div", { class: "card stack" },
        h("h2", null, "Mode"),
        app.debug
          ? notice("warn", "code", h("strong", null, "Developer mode. "), "Scores and reasons are shown after every attempt.")
          : notice("ok", "eyeOff", h("strong", null, "Protected mode. "), "Only decisions are shown."),
        toggle("Developer mode (show scores and reasons)", Boolean(sys.debug), async (on) => {
          await api.setOptions({ debug: on });
          reload(on ? "Developer mode on." : "Protected mode on.");
        }),
        toggle("Liveness check before each sign-in", sys.liveness !== false, async (on) => {
          await api.setOptions({ liveness: on });
          reload(on ? "Liveness check on." : "Liveness check off (demonstrations only).");
        }),
        h("p", { class: "small muted" }, "Anyone holding the unlocked phone can change these; see Security, Known limitations.")),
      h("div", { class: "card stack" },
        h("h2", null, "Storage"),
        storageText,
        h("div", { class: "row" }, persistBtn)),
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", null, "About")),
        h("dl", { class: "kv" },
          h("dt", null, "Runs"), h("dd", null, "Entirely on this device (nothing is uploaded)"),
          h("dt", null, "Python (Pyodide)"), h("dd", null, `${sys.python || "—"} · numpy ${mob.numpy || "—"} · scipy ${mob.scipy || "—"}`),
          h("dt", null, "Models"), h("dd", null, "MediaPipe hand landmarker (web) · YuNet · SFace"),
          h("dt", null, "Recording per attempt"), h("dd", null, sys.recording_sec ? `${sys.recording_sec} s` : "—"),
          h("dt", null, "Samples per registration"), h("dd", null, String(sys.samples_required || "—")),
          h("dt", null, "Minimum hand frames"), h("dd", null, String(sys.min_hand_frames || "—")))));
  },
};
