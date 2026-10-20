// WaveLock mobile: Security (replaces web/js/views/security.js: the
// protections and limits of the on-device app differ from the desktop's).

import { api } from "../api.js";
import { h, icon, clear, toast, notice, fmtTime, fmtSeconds, add } from "../ui.js";

function protection(iconName, title, text, warn = false) {
  return h("div", { class: `card protect ${warn ? "warn" : ""}` },
    h("span", { class: "p-icon" }, icon(iconName)),
    h("div", null, h("h3", null, title), h("p", null, text)));
}

export default {
  id: "security",
  navLabel: "Security",
  title: "Security",
  subtitle: "Protections in force, known limits and the sign-in log.",
  icon: "security",
  camera: false,

  async mount(root, app) {
    const sys = app.system || {};
    const lim = sys.limits || { max_failures: 5, lockout_base_sec: 30, lockout_max_sec: 900 };

    add(root,
      h("div", { class: "section-label" }, "Protections in force"),
      h("div", { class: "grid grid-3" },
        protection("monitor", "On this device only",
          "The camera, the hand and face models and every decision run on this phone. Profiles are stored in this browser's storage for this site and are never uploaded."),
        sys.liveness === false
          ? protection("liveness", "Liveness check is OFF", "Switched off in Settings: sign-in records the gesture without the challenge. Use this for demonstrations only.", true)
          : protection("liveness", "Liveness check",
            "Before each sign-in: show the requested number of fingers (all five count) and, with a face enrolled, turn your head as asked. The face seen must be the enrolled one. Then you start the gesture yourself, within 20 s."),
        protection("face", "Face check fails closed",
          "If a profile has a face enrolled but the face file is missing or damaged, sign-in is refused instead of skipping the face check."),
        protection("lock", "Attempt limit",
          `${lim.max_failures} failed attempts in a row lock the account for ${fmtSeconds(lim.lockout_base_sec)}, ` +
          `doubling with each lockout up to ${fmtSeconds(lim.lockout_max_sec)}.`),
        app.debug
          ? protection("code", "Developer mode is ON", "Scores, cosines and failure reasons are visible. Switch it off in Settings for real sign-ins.", true)
          : protection("eyeOff", "Scores hidden", "Sign-in shows only the decision, so an observer cannot tune an imitation from the scores."),
        protection("scan", "Face over the recording",
          "The face must be found and must match in most of six frames of the recording, not in a single lucky frame."),
        protection("wave", "Registration quality gate",
          `A sample that fails the consistency check ${sys.max_retries_per_sample || 3} times is never accepted; ` +
          `the whole set is recorded again (up to ${sys.max_registration_restarts || 2} restarts).`),
        protection("signin", "Owner-only profile changes",
          "Re-registering a username or enrolling / replacing its face needs a granted sign-in of that user in the last 10 minutes."),
        protection("refresh", "Guarded template aging",
          "A template is replaced only after a very confident sign-in, and only if the gesture still matches the original enrolment; every update is logged.")),

      h("div", { class: "card stack" },
        h("h2", null, "Known limitations"),
        notice("warn", "alert", h("strong", null, "The liveness check is a countermeasure, not a proof. "),
          "It stops a printed photo or a replayed video that cannot follow a random instruction. A live helper next to the owner, or a real-time deepfake that follows the instructions, is not stopped by it. It was tested with simulated input only, not with real attack material."),
        notice("warn", "alert", h("strong", null, "Profiles are stored unencrypted in the browser. "),
          "They are protected by the phone's lock screen and the browser's per-site isolation only (the desktop's Windows encryption does not exist here). Clearing this site's data deletes them."),
        notice("warn", "alert", h("strong", null, "Settings are not protected. "),
          "Anyone holding the unlocked phone can switch on developer mode (which shows the scores) or switch off the liveness check, and can register a new username."),
        notice("warn", "alert", h("strong", null, "Relaxed profiles skip the movement-rhythm check. "),
          "They are rejected less often when the speed varies, but are weaker against someone who copies the gesture hesitantly."),
        notice("info", "info", h("strong", null, "A phone profile is a separate profile. "),
          "The phone uses MediaPipe's web hand model, its own camera and frame rate; desktop profiles are not used here. Register on the phone, and hold it the same way as during registration. The face check uses six frames at evenly spaced times of the recording (the desktop samples six frames by index)."),
        notice("info", "flask", h("strong", null, "Evaluation status. "),
          "The matching code is the desktop's (same Python modules, checked to give identical results). The MEASURED results come from desktop recordings; the phone app has not been measured with real users or real attacks.")));

    const filter = h("select", { class: "select", "aria-label": "Filter by user" }, h("option", { value: "" }, "All users"));
    const tableBox = h("div", { class: "table-wrap" });
    const loadLog = async () => {
      let data;
      try { data = await api.activity(100, filter.value || null); } catch (e) { toast(e.message, "danger"); return; }
      clear(tableBox);
      if (!data.entries.length) {
        add(tableBox, h("div", { class: "empty" }, icon("activity"), h("p", null, "No attempts recorded.")));
        return;
      }
      add(tableBox, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", null, "Time"), h("th", null, "User"), h("th", null, "Result"),
          app.debug ? h("th", null, "Reason") : null)),
        h("tbody", null, data.entries.map((e) => h("tr", null,
          h("td", { class: "muted" }, fmtTime(e.time)),
          h("td", null, h("strong", null, e.user)),
          h("td", null, h("span", { class: `pill sm ${e.granted ? "ok" : "danger"}` }, h("span", { class: "dot" }),
            e.granted ? "Granted" : "Denied")),
          app.debug ? h("td", { class: "mono tiny muted" }, e.reason || "—") : null)))));
    };
    filter.addEventListener("change", loadLog);
    try {
      for (const u of await api.users()) filter.append(h("option", { value: u.name }, u.name));
    } catch (e) { /* the log still loads */ }

    add(root, h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", null, h("h2", null, "Sign-in log"),
          h("div", { class: "sub" }, app.debug ? "Including failure reasons (developer mode)." : "Decisions only; reasons are hidden in protected mode.")),
        h("div", { class: "row" }, filter,
          h("button", { class: "btn sm", type: "button", onclick: loadLog }, icon("refresh"), "Refresh"))),
      tableBox));
    await loadLog();
  },
};
