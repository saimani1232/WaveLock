import { api } from "../api.js";
import { CameraPanel } from "../camera.js";
import { handMark } from "../brand.js";
import { h, icon, clear, toast, initials, notice, fmtSeconds, fmtNum, fill, add } from "../ui.js";

const STAGES = [
  { key: "face", label: "Face anchor", icon: "face" },
  { key: "gesture", label: "Gesture gates", icon: "wave" },
  { key: "hand", label: "Hand biometrics", icon: "bones" },
  { key: "decision", label: "Decision", icon: "decision" },
];

const FAIL_TEXT = {
  distance: "pose DTW", finger_state: "finger states", segment_mismatch: "worst segment",
  transition_order: "finger order", trajectory: "wrist path", low_confidence: "fused score",
  impostor_hand_morphology: "hand anatomy", impostor_kinematic_dynamics: "movement rhythm",
  impostor_face_identity: "face: different person", lookalike_sibling_detected: "face: look-alike",
  no_face_detected: "face: not detected", unstable_face_match: "face: unstable",
  insufficient_face_frames: "face: not in view long enough", too_few_frames: "too few hand frames",
};

function meter(label, value, limit, pass, { higherBetter = false, min = 0, max = null, digits = 3 } = {}) {
  const hasValue = value !== null && value !== undefined;
  let top = max;
  if (top === null) top = (limit || 1) / 0.7;            // limit sits at 70 % of the bar
  const frac = (x) => Math.max(0, Math.min(1, (x - min) / (top - min || 1)));
  const fill = hasValue ? frac(value) : 0;
  const mark = limit !== null && limit !== undefined ? frac(limit) : null;
  return h("div", { class: `meter ${pass ? "" : "fail"}` },
    h("div", { class: "meter-top" },
      h("span", null, pass ? "✓ " : "✕ ", label),
      h("span", { class: "val" }, hasValue ? fmtNum(value, digits) : "—",
        limit !== null && limit !== undefined ? ` ${higherBetter ? "≥" : "≤"} ${fmtNum(limit, digits)}` : "")),
    h("div", { class: "meter-bar" },
      h("div", { class: "meter-fill", vars: { "--w": `${(fill * 100).toFixed(1)}%` } }),
      mark !== null ? h("div", { class: "meter-limit", vars: { "--l": `${(mark * 100).toFixed(1)}%` } }) : null));
}

function detailsCard(details, result) {
  if (!details || !details.gates) {
    return h("div", { class: "card" },
      h("div", { class: "dev-tag" }, icon("code"), "Developer details"),
      h("p", { class: "muted small" }, `Reason: ${details ? details.failure_reason : "—"}`));
  }
  const f = details.face, m = details.macro, hd = details.hand;
  const kin = hd.kin_details || {}, an = hd.anthro_details || {};
  const reasons = (details.failure_reason || "").split("_and_").filter(Boolean)
    .map((r) => FAIL_TEXT[r] || r);
  return h("div", { class: "card stack" },
    h("div", { class: "row between" },
      h("div", { class: "dev-tag" }, icon("code"), "Developer details"),
      h("span", { class: "tiny faint" }, `${details.hand_frames} hand frames · best template ${m.best_template}`)),
    reasons.length ? notice("danger", "alert", h("strong", null, "Failed: "), reasons.join(", ")) : null,
    h("div", { class: "section-label" }, "Face anchor (SFace cosine)"),
    f.enrolled ? meter("Median cosine", f.score, f.threshold, Boolean(f.match),
      { higherBetter: true, min: -0.1, max: 0.9 })
      : h("p", { class: "small muted" }, "No face enrolled (legacy profile)."),
    h("div", { class: "section-label" }, "Gesture gates (lower is better)"),
    details.gates.filter((g) => g.active !== false).map((g) => meter(g.label, g.value, g.limit, Boolean(g.pass))),
    meter("Fused gesture score", m.fused_score, m.threshold, m.fused_score >= m.threshold,
      { higherBetter: true, max: 1 }),
    h("div", { class: "section-label" }, "Hand biometrics"),
    meter(`Anatomy (${an.passing_features || "?"} ratios in range)`, hd.anthro_confidence, null,
      Boolean(hd.anthro_match), { higherBetter: true, max: 1, digits: 2 }),
    meter(`Rhythm · jerk ${fmtNum(kin.jerk_ratio, 2)}× · pause ${kin.max_mid_dwell}/${kin.allowed_dwell}`,
      hd.kin_confidence, null, Boolean(hd.kin_match), { higherBetter: true, max: 1, digits: 2 }),
    hd.soft_consensus_used ? notice("warn", "alert", "Soft consensus used: a strong face match admitted borderline hand anatomy.") : null,
    result.template_updated ? notice("info", "refresh", `Template ${result.template_updated} was replaced (adaptive aging).`) : null,
    h("p", { class: "tiny faint" }, `Multimodal display score ${fmtNum(details.multimodal_fused_score, 3)} (not used for the decision).`));
}

// Static markup (no user data): an animated padlock and a countdown ring.
function lockIcon(open) {
  const span = h("span", { class: `lock-anim ${open ? "opening" : ""}`, "aria-hidden": "true" });
  span.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path class="shackle" d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>' +
    '<rect x="4.5" y="10.5" width="15" height="10.5" rx="2.5"/><path d="M12 15v2"/></svg>';
  return span;
}

function countdownRing(left, total) {
  const span = h("span", { class: "count-ring", "aria-hidden": "true" });
  const frac = total > 0 ? Math.max(0, Math.min(1, left / total)) : 0;
  span.innerHTML = '<svg viewBox="0 0 40 40"><circle cx="20" cy="20" r="16" class="cr-track"/>' +
    `<circle cx="20" cy="20" r="16" class="cr-bar" pathLength="100" stroke-dasharray="100" stroke-dashoffset="${(100 * (1 - frac)).toFixed(1)}"/></svg>`;
  add(span, h("b", null, String(Math.ceil(left))));
  return span;
}

export default {
  id: "signin",
  navLabel: "Sign in",
  title: "Sign in",
  subtitle: "Your face and a hand gesture are checked together on this webcam.",
  icon: "signin",
  camera: true,

  async mount(root, app, params) {
    let cam = null, unsub = null, releasePoll = null, lastKey = "", session = null, busy = false;
    let seenOnServer = false;   // ignore polls that started before this session existed
    let lockTotal = 0;          // longest remaining lock seen, for the countdown ring
    const onKey = (e) => {
      if (!session || session.phase !== "ready") return;
      if (e.target.closest && e.target.closest("input, textarea, select, button")) return;
      if (e.key === "r" || e.key === "R" || e.key === " ") { e.preventDefault(); record(); }
    };
    document.addEventListener("keydown", onKey);

    // ------------------------------------------------------------ landing
    const showLanding = (prefill = "", error = "") => {
      session = null;
      if (cam) { cam.stop(); cam = null; }
      const input = h("input", {
        class: "input", id: "username", autocomplete: "off", spellcheck: "false",
        placeholder: "Your username", value: prefill, "aria-describedby": "user-error",
      });
      const err = h("div", { class: "error-text", id: "user-error", role: "alert" }, error);
      const submit = h("button", { class: "btn primary lg block", type: "submit" }, "Continue", icon("arrowRight"));
      const form = h("form", { class: "stack", novalidate: true, onsubmit: async (e) => {
        e.preventDefault();
        const name = input.value.trim();
        if (!name) { err.textContent = "Enter your username."; input.focus(); return; }
        submit.disabled = true;
        err.textContent = "";
        try {
          const state = await api.authBegin(name);
          showSession(state);
        } catch (ex) {
          err.textContent = ex.code === "not_registered"
            ? "No profile with this username. Check the spelling or register first."
            : ex.message;
          submit.disabled = false;
          input.focus();
        }
      } },
      h("div", { class: "field" }, h("label", { for: "username" }, "Username"), input,
        app.debug ? h("div", { class: "help" }, "Developer mode: see Users for the list of profiles.") : null),
      err, submit);
      fill(root, h("div", { class: "hero-auth" },
        h("div", { class: "card" },
          h("div", { class: "hero-brand" }, handMark({ animated: true, className: "hero-hand" }),
            h("div", null, h("p", { class: "eyebrow" }, "WaveLock sign-in"),
              h("h2", null, "Your hand is your password."))),
          h("p", { class: "muted" }, "Enter your username, then face the camera and perform the gesture only you know."),
          h("div", { class: "divider" }), form,
          h("div", { class: "feature-row" },
            h("div", { class: "feature" }, icon("face"), "Face anchor"),
            h("div", { class: "feature" }, icon("wave"), "Gesture password"),
            h("div", { class: "feature" }, icon("bones"), "Hand biometrics")))));
      setTimeout(() => input.focus(), 50);
    };

    // ------------------------------------------------------------ session
    const pipeline = h("div", { class: "pipeline" });
    const action = h("div", { class: "stack", "aria-live": "polite" });
    const details = h("div");

    const record = async () => {
      if (busy) return;
      if (!cam || !cam.live) { toast("The camera is not ready yet.", "warn"); return; }
      busy = true;
      try {
        await api.authRecord();
        app.poll();
      } catch (e) {
        toast(e.code === "no_hand" ? "No hand detected. Hold your hand in front of the camera, then try again." : e.message,
          e.code === "locked" ? "danger" : "warn");
      } finally {
        busy = false;
      }
    };

    const endSession = async (goHome) => {
      try { await api.authEnd(); } catch (e) { /* already ended */ }
      if (goHome) app.go("dashboard"); else showLanding();
    };

    const showSession = (state) => {
      session = state;
      seenOnServer = false;
      cam = new CameraPanel({ showFace: state.face_enrolled, label: "Live camera for sign-in" });
      fill(root, h("div", { class: "split" },
        h("div", { class: "stack" }, cam.el,
          h("p", { class: "hint-line" }, icon("info"),
            h("span", null, "Press ", h("span", { class: "kbd" }, "R"), " or ", h("span", { class: "kbd" }, "Space"),
              " to record. Keep your hand and face in the frame for the whole recording."))),
        h("div", { class: "stack" },
          h("div", { class: "card" },
            h("div", { class: "user-head" },
              h("span", { class: "avatar lg" }, initials(state.username)),
              h("div", { class: "spacer" },
                h("div", { class: "name" }, state.username),
                h("div", { class: "small muted" }, `${state.templates} gesture templates · ` +
                  (state.face_enrolled ? "face anchor on" : "no face anchor (legacy profile)"))),
              h("button", { class: "btn sm ghost", type: "button", onclick: () => endSession(false) },
                icon("logout"), "Switch user"))),
          h("div", { class: "card stack" }, action,
            h("div", { class: "section-label" }, "Verification pipeline"), pipeline),
          details)));
      cam.setCoach({ needHand: true, needFace: state.face_enrolled, text: "Ready when you are" });
      cam.start(app);
      lastKey = "";
      render(state);
    };

    const renderPipeline = (s) => {
      const r = s.result;
      const d = r && r.details && r.details.gates ? r.details : null;
      const stateOf = (key) => {
        if (key === "face" && !s.face_enrolled) return "off";
        if (s.phase === "recording") return key === "face" || key === "gesture" ? "active" : "";
        if (s.phase === "processing") return "active";
        if (s.phase === "result" && r) {
          if (key === "decision") return r.granted ? "pass" : "fail";
          if (r.too_few_frames) return "";
          // Protected mode reveals no per-stage verdict: the stages only say
          // they were checked, and only the decision is coloured.
          if (!d) return "checked";
          if (key === "face") return d.face.match ? "pass" : "fail";
          if (key === "gesture") return d.macro.passed ? "pass" : "fail";
          if (key === "hand") return d.hand.passed ? "pass" : "fail";
        }
        return "";
      };
      fill(pipeline, STAGES.map((st, i) => {
        const cls = stateOf(st.key);
        const iconName = cls === "pass" ? "check" : cls === "fail" ? "x" : st.icon;
        const note = { pass: "Passed", fail: "Failed", checked: "Checked", off: "Not enrolled", active: "Checking" }[cls] || "";
        return h("div", { class: `stage ${cls}`, vars: { "--i": String(i) } },
          h("span", { class: "stage-icon" }, icon(iconName)), st.label,
          note ? h("span", { class: "stage-note" }, note) : null);
      }));
      if (s.phase === "result" && r && !d && !r.too_few_frames) {
        add(pipeline, h("p", { class: "pipeline-note" },
          "Protected mode does not say which check failed, so an observer cannot tune an imitation."));
      }
    };

    const renderAction = (s) => {
      clear(action);
      clear(details);
      const r = s.result;
      if (s.error) add(action, notice("warn", "alert", s.error));
      if (s.phase === "ready") {
        if (s.locked_for > 0) {
          lockTotal = Math.max(lockTotal, s.locked_for);
          add(action, h("div", { class: "decision warn" },
            h("span", { class: "d-icon ring-icon" }, countdownRing(s.locked_for, lockTotal)),
            h("div", null, h("h3", null, "Temporarily locked"),
              h("p", { class: "small muted" }, `Too many failed attempts. Try again in ${fmtSeconds(s.locked_for)}.`))));
        } else {
          lockTotal = 0;
          add(action, h("ol", { class: "tips" },
            h("li", null, "Face the camera so your face is visible."),
            h("li", null, "Show your hand until “Hand detected” appears."),
            h("li", null, "Start recording and perform your gesture at your normal speed.")));
        }
        add(action, h("button", {
          class: "btn primary lg block", type: "button",
          disabled: s.locked_for > 0 || !(cam && cam.live), onclick: record,
        }, icon("record"), cam && cam.live ? "Start recording" : "Waiting for the camera…"));
        cam && cam.setMode("idle", { hint: s.locked_for > 0 ? "Locked: wait for the timer" : "Ready: press R to record" });
      } else if (s.phase === "recording") {
        add(action, h("div", { class: "decision warn" }, h("span", { class: "d-icon" }, icon("wave")),
          h("div", null, h("h3", null, "Perform your gesture now"),
            h("p", { class: "small muted" }, "Keep going until the ring completes."))));
        cam && cam.setMode("recording", s.recording || {});
      } else if (s.phase === "processing") {
        add(action, h("div", { class: "decision" }, h("div", { class: "spinner sm" }),
          h("div", null, h("h3", null, "Verifying"),
            h("p", { class: "small muted" }, "Checking face, gesture gates and hand biometrics…"))));
        cam && cam.setMode("processing", { text: "Verifying identity…" });
      } else if (s.phase === "result" && r) {
        if (r.too_few_frames) {
          add(action, h("div", { class: "decision warn" }, h("span", { class: "d-icon" }, icon("hand")),
            h("div", null, h("h3", null, "Hand not seen long enough"),
              h("p", { class: "small muted" }, `Only ${r.hand_frames} hand frames were captured. This attempt was not counted.`))));
        } else if (r.granted) {
          add(action, h("div", { class: "decision ok" }, h("span", { class: "d-icon" }, lockIcon(true)),
            h("div", null, h("h3", null, "Access granted"),
              h("p", { class: "small muted" }, `Welcome, ${s.username}. All checks agreed.`))));
        } else {
          add(action, h("div", { class: "decision danger shake" }, h("span", { class: "d-icon" }, lockIcon(false)),
            h("div", null, h("h3", null, "Access denied"),
              h("p", { class: "small muted" }, r.locked_for > 0
                ? `Too many failed attempts. Locked for ${fmtSeconds(r.locked_for)}.`
                : app.debug ? "See the developer details below." :
                  `The checks did not agree. ${r.attempts_left} attempt${r.attempts_left === 1 ? "" : "s"} left before a temporary lock.`))));
        }
        if (r.too_few_frames) cam && cam.setMode("idle", { hint: "Keep your hand in view, then try again" });
        else if (r.granted) cam && cam.setMode("granted", { title: "Access granted", sub: `Welcome, ${s.username}` });
        else cam && cam.setMode("denied", { title: "Access denied",
          sub: r.locked_for > 0 ? "Temporarily locked" : `${r.attempts_left} attempt${r.attempts_left === 1 ? "" : "s"} left` });
        add(action, h("div", { class: "row" },
          h("button", { class: "btn primary", type: "button", onclick: async () => {
            try { await api.authReset(); app.poll(); } catch (e) { toast(e.message, "danger"); }
          } }, icon("refresh"), r.granted ? "Sign in again" : "Try again"),
          h("button", { class: "btn ghost", type: "button", onclick: () => endSession(true) }, "Done")));
        if (app.debug && r.details) add(details, detailsCard(r.details, r));
      }
    };

    const render = (s) => {
      session = s;
      renderPipeline(s);
      renderAction(s);
    };

    unsub = app.subscribe((st) => {
      if (cam) cam.update(st.camera);
      const s = st.auth;
      if (!session) return;
      if (!s) { if (seenOnServer) showLanding(); return; }
      if (s.username !== session.username) return;
      seenOnServer = true;
      const key = `${s.phase}|${s.attempt}|${s.error}|${Math.ceil(s.locked_for)}|${cam && cam.live}`;
      if (key !== lastKey) { lastKey = key; render(s); }
      if (s.phase === "recording" && s.recording && cam) cam.setProgress(s.recording);
    });
    releasePoll = app.needFastPoll();

    // Resume an existing session (page reload) or start fresh.
    let current = null;
    try { current = (await api.status()).auth; } catch (e) { /* offline toast elsewhere */ }
    if (current) showSession(current);
    else showLanding(params.get("user") || "");

    return async () => {
      document.removeEventListener("keydown", onKey);
      if (unsub) unsub();
      if (releasePoll) releasePoll();
      if (cam) cam.stop();
      if (session) { try { await api.authEnd(); } catch (e) { /* ignore */ } }
    };
  },
};
