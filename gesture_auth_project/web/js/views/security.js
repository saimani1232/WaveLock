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

    // ------------------------------------------------------------ protections
    add(root, 
      h("div", { class: "section-label" }, "Protections in force"),
      h("div", { class: "grid grid-3" },
        protection("face", "Face check fails closed",
          "If a profile has a face enrolled but the face file is missing or damaged, sign-in is refused instead of skipping the face check."),
        protection("lock", "Attempt limit",
          `${lim.max_failures} failed attempts in a row lock the account for ${fmtSeconds(lim.lockout_base_sec)}, ` +
          `doubling with each lockout up to ${fmtSeconds(lim.lockout_max_sec)}. Shared with the desktop programs.`),
        app.debug
          ? protection("code", "Developer mode is ON", "Per-gate scores, cosines and failure reasons are visible. Restart without --debug for real sign-ins.", true)
          : protection("eyeOff", "Scores hidden", "Sign-in shows only the decision, so an observer cannot tune an imitation from the scores."),
        protection("scan", "Face over most frames",
          "The face must be found and must match in most sampled frames of the recording, not in a single lucky frame."),
        protection("wave", "Registration quality gate",
          `A sample that fails the consistency check ${sys.max_retries_per_sample || 3} times is never accepted; ` +
          `the whole set is recorded again (up to ${sys.max_registration_restarts || 2} restarts).`),
        protection("monitor", "Local-only server",
          "Listens on 127.0.0.1 only, rejects foreign hosts and cross-site requests, and serves a strict content security policy."),
        sys.liveness === false
          ? protection("liveness", "Liveness check is OFF", "The server was started with --no-liveness: sign-in records the gesture without the challenge. Use this for demonstrations only.", true)
          : protection("liveness", "Liveness check",
            "Before each sign-in: show the requested number of fingers (all five count) and, with a face enrolled, turn your head as asked. The face seen must be the enrolled one. Then you start the gesture yourself, within 20 s."),
        protection("signin", "Owner-only profile changes",
          "Re-registering a username or enrolling / replacing its face needs a granted sign-in of that user in the last 10 minutes."),
        protection("lock", "Profiles encrypted at rest",
          "New templates, recordings and face embeddings are encrypted with Windows DPAPI, readable only by this Windows account on this computer."),
        protection("refresh", "Guarded template aging",
          "A template is replaced only after a very confident sign-in, and only if the gesture still matches the original enrolment (drift anchor); every update is logged."),
        protection("users", "No public user list",
          "The sign-in screens never list usernames (only developer mode shows them).")),

      h("div", { class: "card stack" },
        h("h2", null, "Known limitations"),
        notice("warn", "alert", h("strong", null, "The liveness check is a countermeasure, not a proof. "),
          "It stops a printed photo or a replayed video that cannot follow a random instruction. A live helper next to the owner, or a real-time deepfake that follows the instructions, is not stopped by it. It was tested with simulated cameras only, not with real attack material."),
        notice("warn", "alert", h("strong", null, "New usernames are open. "),
          "Anyone at this computer can register a new username; only changes to an existing profile need the owner's sign-in."),
        notice("warn", "alert", h("strong", null, "Relaxed profiles skip the movement-rhythm check. "),
          "They are rejected less often when the speed varies, but are weaker against someone who copies the gesture hesitantly. Face, liveness, gesture and hand anatomy still apply. The Users page shows each profile's check level."),
        notice("warn", "alert", h("strong", null, "Encryption covers this Windows account only. "),
          "Programs running as the same Windows user can read the profiles; config.json (thresholds) stays plain, and profiles created before encryption stay plain until converted with migrate_profiles.py --encrypt."),
        notice("info", "info", h("strong", null, "Start the gesture as during registration. "),
          "Still frames at the start and end of a recording are ignored (new and recalibrated profiles), but a gesture begun before the recording starts loses its beginning and is usually rejected."),
        notice("info", "flask", h("strong", null, "Evaluation status. "),
          "Attack results are SIMULATED (synthetic attackers, perfect replay). Real enrolment recordings give genuine acceptance and zero-effort rejection only, from one recording session per person and a few people (evaluate_real_recordings.py); imitation, photo and replay attacks by real people have not been recorded.")));

    // ------------------------------------------------------------ activity log
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

    // ------------------------------------------------------------ benchmarks (developer mode)
    let benchTimer = null;
    if (app.debug) {
      const userSel = h("select", { class: "select", "aria-label": "User to benchmark" });
      for (const o of filter.querySelectorAll("option")) if (o.value) userSel.append(h("option", { value: o.value }, o.value));
      const kindSel = h("select", { class: "select", "aria-label": "Benchmark" },
        h("option", { value: "gesture" }, "Gesture + hand (150 trials)"),
        h("option", { value: "multimodal" }, "Multimodal (180 trials)"));
      const out = h("pre", { class: "output", "aria-live": "polite" }, "No benchmark run yet.");
      const runBtn = h("button", { class: "btn primary", type: "button" }, icon("flask"), "Run benchmark");
      const show = (b) => {
        if (!b) return;
        if (b.state === "running") {
          runBtn.disabled = true;
          out.textContent = `Running ${b.kind} benchmark for ${b.user}… this takes a minute.`;
        } else {
          runBtn.disabled = false;
          out.textContent = b.output || "(no output)";
          clearInterval(benchTimer);
          benchTimer = null;
        }
      };
      const watch = () => {
        clearInterval(benchTimer);
        benchTimer = setInterval(async () => {
          try { show(await api.benchmarkStatus()); } catch (e) { clearInterval(benchTimer); }
        }, 1500);
      };
      runBtn.addEventListener("click", async () => {
        if (!userSel.value) { toast("Register a user first.", "warn"); return; }
        try { show(await api.benchmarkRun(userSel.value, kindSel.value)); watch(); }
        catch (e) { toast(e.message, "danger"); }
      });
      add(root, h("div", { class: "card stack" },
        h("div", { class: "card-head" },
          h("div", null, h("div", { class: "dev-tag" }, icon("code"), "Developer mode"), h("h2", null, "SIMULATED benchmarks"),
            h("div", { class: "sub" }, "Synthetic attackers built from the user's own templates, reported per attack. Not real-world error rates."))),
        h("div", { class: "row" }, userSel, kindSel, runBtn), out));
      try {
        const b = await api.benchmarkStatus();
        show(b);
        if (b && b.state === "running") watch();
      } catch (e) { /* none yet */ }
    }
    return () => { if (benchTimer) clearInterval(benchTimer); };
  },
};
