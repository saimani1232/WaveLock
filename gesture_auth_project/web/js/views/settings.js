import { api } from "../api.js";
import { h, icon, clear, toast, notice, fill, add } from "../ui.js";

export default {
  id: "settings",
  navLabel: "Settings",
  title: "Settings",
  subtitle: "Appearance, camera and system information.",
  icon: "settings",
  camera: false,

  async mount(root, app) {
    const sys = app.system || {};

    // ------------------------------------------------------------ appearance
    const pref = app.themePreference();
    const seg = h("div", { class: "segmented", role: "group", "aria-label": "Theme" });
    const options = [["light", "sun", "Light"], ["dark", "moon", "Dark"], ["system", "monitor", "System"]];
    for (const [value, ic, label] of options) {
      seg.append(h("button", { type: "button", "aria-pressed": String(pref === value), onclick: (e) => {
        app.setTheme(value);
        for (const b of seg.children) b.setAttribute("aria-pressed", "false");
        e.currentTarget.setAttribute("aria-pressed", "true");
      } }, icon(ic), label));
    }

    // ------------------------------------------------------------ camera
    const camState = h("p", { class: "small muted" });
    const list = h("div", { class: "list" });
    const perf = h("p", { class: "small muted" });
    const paintCam = (st) => {
      const c = st && st.camera;
      if (!c) return;
      camState.textContent = c.state === "running"
        ? `Running (camera ${c.camera_index === null ? "auto-detected" : c.camera_index}, ${Math.round(c.fps)} fps).`
        : c.state === "error" ? `Error: ${c.error}` : "Stopped. It starts automatically on the Sign in and Register pages.";
      // Frame-rate diagnosis: time waiting for the webcam vs. processing time.
      if (c.state === "running" && c.process_ms > 0) {
        const capacity = Math.floor(1000 / c.process_ms);
        const cameraBound = c.read_ms >= c.process_ms;
        perf.textContent = `Per frame: ${Math.round(c.read_ms)} ms waiting for the webcam, ` +
          `${Math.round(c.process_ms)} ms processing (hand tracking, face box, image) — enough for about ${capacity} fps. ` +
          (cameraBound
            ? "The webcam sets the pace: it delivers frames at this rate (often 15 fps indoors for built-in cameras; more light helps)."
            : "Processing sets the pace: close other heavy programs, or plug in the laptop charger.");
      } else {
        perf.textContent = "";
      }
    };
    const unsub = app.subscribe(paintCam);
    const scan = async () => {
      fill(list, h("div", { class: "row small muted" }, h("span", { class: "spinner sm" }), "Scanning cameras…"));
      try {
        const cams = await api.cameras();
        clear(list);
        if (!cams.length) add(list, h("div", { class: "empty" }, icon("cameraOff"), h("p", null, "No camera found.")));
        for (const c of cams) {
          add(list, h("div", { class: "list-item" },
            h("span", { class: `status-icon ${c.status === "READY" ? "ok" : "neutral"}` }, icon("camera")),
            h("div", { class: "spacer" }, h("strong", null, `Camera ${c.index}`),
              h("div", { class: "tiny muted" }, `${c.width}×${c.height} · ${c.backend} · ${c.status}`)),
            h("button", { class: "btn sm", type: "button", onclick: async () => {
              try {
                const res = await api.cameraStart(c.index);
                if (res.state === "running") toast(`Camera ${c.index} works and will be used from now on.`, "ok");
                else toast(res.error || "Camera did not start.", "danger");
                await api.cameraStop();
                app.poll();
              } catch (e) { toast(e.message, "danger"); }
            } }, "Use this camera")));
        }
      } catch (e) {
        fill(list, notice("warn", "alert", e.message));
      }
    };

    add(root, 
      h("div", { class: "grid grid-2" },
        h("div", { class: "card stack" },
          h("h2", null, "Appearance"),
          h("p", { class: "small muted" }, "Light, dark, or follow the operating system. Saved in this browser."),
          seg),
        h("div", { class: "card stack" },
          h("h2", null, "Mode"),
          app.debug
            ? notice("warn", "code", h("strong", null, "Developer mode. "), "Started with --debug: scores, reasons and benchmarks are visible.")
            : notice("ok", "eyeOff", h("strong", null, "Protected mode. "), "Only decisions are shown."),
          h("p", { class: "small muted" }, "The mode is chosen when the server starts (", h("code", null, "python web_app.py --debug"),
            "); it cannot be switched from the browser."))),
      h("div", { class: "card stack" },
        h("div", { class: "card-head" },
          h("div", null, h("h2", null, "Camera"), camState, perf),
          h("div", { class: "row" },
            h("button", { class: "btn sm", type: "button", onclick: async () => {
              try { await api.cameraStop(); app.poll(); } catch (e) { toast(e.message, "warn"); }
            } }, icon("cameraOff"), "Stop camera"),
            h("button", { class: "btn sm primary", type: "button", onclick: scan }, icon("refresh"), "Scan devices"))),
        h("p", { class: "small muted" }, "To always use a specific camera, start the server with ",
          h("code", null, "--camera N"), ". Scanning opens each device briefly, so the camera must be stopped first."),
        list),
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", null, "About")),
        h("dl", { class: "kv" },
          h("dt", null, "Python"), h("dd", null, sys.python || "—"),
          h("dt", null, "Face models"), h("dd", null, sys.models ? `YuNet ${sys.models.yunet ? "✓" : "✕"} · SFace ${sys.models.sface ? "✓" : "✕"}` : "—"),
          h("dt", null, "Recording per attempt"), h("dd", null, sys.recording_sec ? `${sys.recording_sec} s` : "—"),
          h("dt", null, "Samples per registration"), h("dd", null, String(sys.samples_required || "—")),
          h("dt", null, "Minimum hand frames"), h("dd", null, String(sys.min_hand_frames || "—")),
          h("dt", null, "Server"), h("dd", null, location.host))));
    return () => unsub();
  },
};
