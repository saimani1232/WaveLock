import { api } from "../api.js";
import { CameraPanel } from "../camera.js";
import { h, icon, clear, toast, notice, confirmDialog, fmtNum, poseLabel, fill, useShortcuts, kbdHint } from "../ui.js";

const STEPS = [
  { key: "account", label: "Account" },
  { key: "face", label: "Face anchor" },
  { key: "gesture", label: "Gesture samples" },
  { key: "review", label: "Review" },
];

function stepper(current) {
  const idx = STEPS.findIndex((s) => s.key === current);
  const parts = [];
  STEPS.forEach((s, i) => {
    if (i) parts.push(h("span", { class: "step-sep", "aria-hidden": "true" }));
    const cls = i < idx ? "done" : i === idx ? "current" : "";
    parts.push(h("div", { class: `step ${cls}`, "aria-current": i === idx ? "step" : null },
      h("span", { class: "num" }, i < idx ? icon("check") : String(i + 1)),
      h("span", { class: "label" }, s.label)));
  });
  return h("nav", { class: "stepper", "aria-label": "Registration steps" }, parts);
}

function eventView(e) {
  const map = {
    accepted: ["ok", "check", `Sample ${e.sample} accepted.`],
    rejected: ["warn", "alert", `Sample ${e.sample} did not match the earlier ones ` +
      `(${e.rejections}/${e.max_retries}). Record it again the same way.`],
    too_few_frames: ["warn", "hand", e.message],
    restart: ["danger", "refresh", `Samples stayed inconsistent, so the whole set is recorded again ` +
      `(restart ${e.restart}/${e.max_restarts}).`],
    error: ["danger", "alert", e.message],
  };
  const [kind, ic, text] = map[e.type] || ["info", "info", e.message || e.type];
  return notice(kind, ic, text);
}

export default {
  id: "register",
  navLabel: "Register",
  title: "Register a user",
  subtitle: "Enrol a face anchor and five samples of a personal hand gesture.",
  icon: "register",
  camera: true,

  async mount(root, app, params) {
    let cam = null, unsub = null, releasePoll = null, reg = null, lastKey = "", busy = false;
    let suppress = false;       // after closing, ignore stale polls until the server reports no session

    const ensureCam = (showFace) => {
      if (!cam) {
        cam = new CameraPanel({ showFace, label: "Live camera for registration" });
        cam.start(app);
      }
      return cam;
    };
    const dropCam = () => { if (cam) { cam.stop(); cam = null; } };

    const act = async (fn, okMsg) => {
      if (busy) return;
      busy = true;
      try {
        await fn();
        if (okMsg) toast(okMsg, "ok");
        app.poll();
      } catch (e) {
        toast(e.code === "no_hand" ? "No hand detected. Hold your hand in view, then record." : e.message,
          e.code === "no_hand" ? "warn" : "danger");
      } finally {
        busy = false;
      }
    };

    const cancel = async () => {
      if (reg && reg.status === "active") {
        const ok = await confirmDialog({ title: "Cancel registration?",
          message: "Nothing is saved. Any existing profile with this name stays unchanged.",
          confirmLabel: "Cancel registration", danger: true });
        if (!ok) return;
        try { await api.registerCancel(); } catch (e) { /* already ended */ }
      }
      suppress = true;
      try { await api.registerClose(); } catch (e) { /* ignore */ }
      reg = null;
      showAccount();
    };

    // Keyboard shortcuts that apply in the current step (see useShortcuts).
    const shortcuts = () => {
      const r = reg;
      if (!r) return [];
      const live = !!(cam && cam.live);
      const list = [];
      if (r.status === "active" && r.step === "face" && r.face_status !== "capturing") {
        if (live) list.push({ keys: ["c"], label: "Capture face", run: () => act(api.registerFaceCapture) });
        list.push({ keys: ["s"], label: "Skip the face (gesture only)", run: () => act(api.registerFaceSkip) });
      }
      if (r.status === "active" && r.step === "gesture" && r.awaiting_record && live) {
        list.push({ keys: ["r", "space"], label: "Record the next sample", run: () => act(api.registerRecord) });
      }
      if (r.status === "active") {
        list.push({ keys: ["escape"], label: "Cancel registration (asks first)", run: cancel });
      } else if (r.status === "done" && r.summary) {
        list.push({ keys: ["s"], label: "Sign in now", run: () => app.go(`signin?user=${encodeURIComponent(r.summary.username)}`) });
        list.push({ keys: ["n"], label: "Register another user", run: cancel });
      } else {
        list.push({ keys: ["n"], label: "Start over", run: cancel });
      }
      return list;
    };
    const releaseKeys = useShortcuts(shortcuts);

    // ------------------------------------------------------------ step 1: account
    const showAccount = () => {
      dropCam();
      lastKey = "";
      let pose = "legacy", level = "standard";
      const input = h("input", { class: "input", id: "reg-user", autocomplete: "off", spellcheck: "false",
        placeholder: "letters, digits, _ - .", value: params.get("user") || "" });
      const err = h("div", { class: "error-text", role: "alert" });
      const choice = (value, title, text) => h("button", {
        type: "button", "aria-pressed": String(value === pose),
        onclick: (e) => {
          pose = value;
          for (const b of choiceBox.children) b.setAttribute("aria-pressed", "false");
          e.currentTarget.setAttribute("aria-pressed", "true");
        },
      }, h("strong", null, title), h("span", null, text));
      const choiceBox = h("div", { class: "choice", role: "group", "aria-label": "Pose normalisation" },
        choice("legacy", "Standard", "The method used by all existing profiles (default)."),
        choice("aspect", "Aspect-corrected (experimental)", "Hand ratios stay stable when the hand tilts. Under evaluation in semester 2."));
      const levelChoice = (value, title, text) => h("button", {
        type: "button", "aria-pressed": String(value === level),
        onclick: (e) => {
          level = value;
          for (const b of levelBox.children) b.setAttribute("aria-pressed", "false");
          e.currentTarget.setAttribute("aria-pressed", "true");
        },
      }, h("strong", null, title), h("span", null, text));
      const levelBox = h("div", { class: "choice", role: "group", "aria-label": "Check level" },
        levelChoice("standard", "Standard (recommended)", "Every check, including your movement rhythm (kinematics), which rejects people who copy your gesture hesitantly."),
        levelChoice("relaxed", "Relaxed", "No movement-rhythm check: fewer rejections if your speed varies, but weaker against someone who copies your gesture. Face, liveness, gesture and hand anatomy still apply."));
      const submit = h("button", { class: "btn primary lg", type: "submit" }, "Start enrolment", icon("arrowRight"));
      const form = h("form", { class: "stack", novalidate: true, onsubmit: async (e) => {
        e.preventDefault();
        const name = input.value.trim();
        if (!name) { err.textContent = "Choose a username."; input.focus(); return; }
        submit.disabled = true;
        err.textContent = "";
        try {
          reg = await api.registerBegin(name, false, pose, level);
        } catch (ex) {
          if (ex.code === "exists") {
            const ok = await confirmDialog({ title: `Replace “${name}”?`,
              message: "A profile with this username exists. Its templates and thresholds are replaced " +
                "when the new registration is saved. The current face is kept unless you capture a new one.",
              confirmLabel: "Re-register", danger: true });
            if (ok) {
              try {
                reg = await api.registerBegin(name, true, pose, level);
              } catch (ex2) {
                if (ex2.code === "owner_signin_required") {
                  // only the owner may replace a profile: offer the sign-in
                  fill(err, h("span", null, ex2.message, " "),
                    h("a", { class: "btn sm primary", href: `#/signin?user=${encodeURIComponent(name)}` },
                      icon("signin"), `Sign in as ${name}`));
                } else {
                  err.textContent = ex2.message;
                }
              }
            }
          } else {
            err.textContent = ex.message;
          }
        }
        submit.disabled = false;
        if (reg) { suppress = false; lastKey = ""; render(reg); }
      } },
      h("div", { class: "field" }, h("label", { for: "reg-user" }, "Username"), input,
        h("div", { class: "help" }, "Stored in lower case; spaces become underscores.")),
      h("div", { class: "field" }, h("label", null, "Check level"), levelBox),
      h("div", { class: "field" }, h("label", null, "Pose normalisation"), choiceBox),
      err, h("div", { class: "row" }, submit));

      fill(root, stepper("account"),
        h("div", { class: "split" },
          h("div", { class: "card" }, h("div", { class: "card-head" },
            h("div", null, h("h2", null, "Create a profile"),
              h("div", { class: "sub" }, "About two minutes. You will need good light and one free hand."))), form),
          h("div", { class: "card stack" },
            h("h2", null, "What gets enrolled"),
            notice("info", "face", h("strong", null, "Face anchor. "), "Five frames of your face, about 0.35 s apart (move your head a little), combined into one SFace embedding; refused unless all five show the same person. Optional, but once enrolled it is checked at every sign-in."),
            notice("info", "wave", h("strong", null, "Gesture password. "), "Five 3-second recordings of the same gesture. A finger sequence such as 1-2-4-3 makes the order gate effective."),
            notice("info", "bones", h("strong", null, "Hand biometrics. "), "Bone-length ratios and movement rhythm are calibrated from the same recordings (the rhythm is not used for decisions on a Relaxed profile)."))));
      setTimeout(() => input.focus(), 50);
    };

    // ------------------------------------------------------------ steps 2-4
    const faceStep = (r) => {
      const c = ensureCam(true);
      const dots = h("div", { class: "face-dots", "aria-label": `${r.face_progress} of ${r.face_required} face frames` },
        Array.from({ length: r.face_required }, (_, i) => h("span", { class: i < r.face_progress ? "on" : "" })));
      const capturing = r.face_status === "capturing";
      c.setCoach({ needFace: true, needHand: false, text: capturing ? "Hold still…" : "Centre your face in the frame" });
      c.setMode("idle");
      return h("div", { class: "split" }, c.el,
        h("div", { class: "card stack" },
          h("h2", null, "Face anchor"),
          h("p", { class: "muted" }, "Look straight at the camera in even light. Remove sunglasses or a mask."),
          h("div", { class: "row" }, dots, h("span", { class: "small muted" }, `${r.face_progress}/${r.face_required} frames`)),
          r.face_status === "failed" ? notice("danger", "alert", r.message || "Face capture failed.") : null,
          r.replacing ? notice("info", "info", "Skipping keeps the face already enrolled for this user.") : null,
          h("div", { class: "row" },
            h("button", { class: "btn primary", type: "button", disabled: capturing || !c.live,
              "aria-keyshortcuts": "C", onclick: () => act(api.registerFaceCapture) },
            capturing ? h("span", { class: "spinner sm" }) : icon("face"),
            capturing ? "Capturing…" : c.live ? "Capture face" : "Waiting for the camera…",
            !capturing && c.live ? kbdHint("C") : null),
            h("button", { class: "btn ghost", type: "button", disabled: capturing,
              "aria-keyshortcuts": "S", onclick: () => act(api.registerFaceSkip) },
            "Skip (gesture only)", capturing ? null : kbdHint("S")))));
    };

    const gestureStep = (r) => {
      const c = ensureCam(false);
      const recording = r.phase === "recording";
      c.setCoach({ needHand: true, needFace: false,
        text: r.awaiting_record ? `Ready: press R to record sample ${Math.min(r.current_sample, r.samples_required)}` : "Please wait" });
      if (recording) c.setMode("recording", r.recording || {});
      else if (r.phase === "checking") c.setMode("processing", { text: "Checking consistency…" });
      else c.setMode("idle");
      const samples = h("div", { class: "samples" }, Array.from({ length: r.samples_required }, (_, i) => {
        const n = i + 1;
        let cls = "";
        if (n <= r.accepted) cls = "accepted";
        else if (n === r.current_sample) cls = recording ? "recording" : "current";
        return h("div", { class: `sample ${cls}`, "aria-label": `Sample ${n}` },
          n <= r.accepted ? icon("check") : String(n));
      }));
      const events = r.events.slice().reverse().slice(0, 4).map(eventView);
      return h("div", { class: "split" }, c.el,
        h("div", { class: "card stack" },
          h("div", { class: "row between" },
            h("h2", null, `Sample ${Math.min(r.current_sample, r.samples_required)} of ${r.samples_required}`),
            r.restarts ? h("span", { class: "pill sm warn" }, `Restart ${r.restarts}/${r.max_restarts}`) : null),
          samples,
          h("p", { class: "muted small" }, recording ? "Perform your gesture now." :
            r.phase === "checking" ? "Comparing this sample with the earlier ones…" :
              "Perform exactly the same gesture each time, at your natural speed."),
          h("button", { class: "btn primary lg block", type: "button", disabled: !r.awaiting_record || !c.live,
            "aria-keyshortcuts": "R Space", onclick: () => act(api.registerRecord) },
          recording ? h("span", { class: "spinner sm" }) : icon("record"),
          recording ? "Recording…" : !c.live ? "Waiting for the camera…"
            : `Record sample ${Math.min(r.current_sample, r.samples_required)}`,
          r.awaiting_record && c.live ? kbdHint("R") : null),
          h("p", { class: "hint-line" }, icon("info"), h("span", null,
            h("span", { class: "kbd" }, "R"), " or ", h("span", { class: "kbd" }, "Space"), " records, ",
            h("span", { class: "kbd" }, "Esc"), " cancels, ", h("span", { class: "kbd" }, "?"), " shows all shortcuts.")),
          events.length ? h("div", { class: "event-log", "aria-live": "polite" }, events) : null));
    };

    const reviewStep = (r) => {
      if (cam) app.releaseCamera();      // nothing left to record
      dropCam();
      if (r.status !== "done" || !r.summary) {
        return h("div", { class: "card center stack" },
          h("div", { class: "row", }, h("span", { class: "result-burst danger" }, icon("x"))),
          h("h2", null, r.status === "cancelled" ? "Registration cancelled" : "Registration not saved"),
          h("p", { class: "muted" }, r.message || ""),
          h("div", { class: "row" }, h("button", { class: "btn primary", type: "button", "aria-keyshortcuts": "N",
            onclick: cancel }, "Start over", kbdHint("N"))));
      }
      const s = r.summary, t = s.thresholds;
      const faceText = { enrolled: "Enrolled", kept: "Kept the existing face", skipped: "Skipped (gesture only)" }[s.face];
      return h("div", { class: "split" },
        h("div", { class: "card stack" },
          h("div", { class: "row" }, h("span", { class: "result-burst ok" }, icon("check")),
            h("div", null, h("h2", null, `Registration complete for ${s.username}`),
              h("p", { class: "muted" }, "Thresholds were calibrated from your own samples."))),
          s.warnings.map((w) => notice("warn", "alert", w)),
          h("dl", { class: "kv" },
            h("dt", null, "Samples"), h("dd", null, String(s.samples)),
            h("dt", null, "Face anchor"), h("dd", null, faceText),
            h("dt", null, "Consistency"), h("dd", null, `${fmtNum(s.consistency, 1)} / 100`),
            h("dt", null, "Largest variation"), h("dd", null, fmtNum(s.max_variation, 3)),
            h("dt", null, "Uniqueness vs. common gestures"), h("dd", null,
              s.uniqueness_ratio === null ? "—" : `${fmtNum(s.uniqueness_ratio, 2)}× ${s.unique ? "✓" : "(low)"}`),
            h("dt", null, "Pose method"), h("dd", null, poseLabel(s.pose_method)),
            h("dt", null, "Check level"), h("dd", null, s.check_level === "relaxed" ? "Relaxed (no movement-rhythm check)" : "Standard (all checks)")),
          h("div", { class: "row" },
            h("a", { class: "btn primary", href: `#/signin?user=${encodeURIComponent(s.username)}`, "aria-keyshortcuts": "S" },
              icon("signin"), "Sign in now", kbdHint("S")),
            h("button", { class: "btn ghost", type: "button", "aria-keyshortcuts": "N", onclick: cancel },
              "Register another", kbdHint("N")))),
        h("div", { class: "card stack" },
          h("h2", null, "Calibrated limits"),
          h("p", { class: "small muted" }, "Shown once, to the person who just enrolled. Sign-in screens never show them in protected mode."),
          h("div", { class: "thresholds" },
            [["Pose DTW", t.dtw], ["Finger states", t.finger_state], ["Finger order", t.transition],
              ["Worst segment", t.segment], ["Wrist path", t.trajectory], ["Anatomy tol.", t.anthro_tolerance]]
              .map(([k, v]) => h("div", null, h("span", null, k), h("b", null, v === null || v === undefined ? "off" : fmtNum(v, 3)))))));
    };

    const render = (r) => {
      reg = r;
      const key = JSON.stringify([r.step, r.status, r.face_status, r.face_progress, r.current_sample,
        r.accepted, r.awaiting_record, r.phase, r.restarts, r.events.length, r.message, cam ? cam.live : false]);
      if (key === lastKey) {
        if (r.phase === "recording" && r.recording && cam) cam.setProgress(r.recording);
        return;
      }
      lastKey = key;
      const step = r.status === "active" ? (r.phase === "saving" ? "review" : r.step) : "review";
      let content;
      if (r.status === "active" && r.phase === "saving") {
        dropCam();
        content = h("div", { class: "card center stack" }, h("div", { class: "row" }, h("div", { class: "spinner sm" }),
          h("h2", null, "Calibrating thresholds and saving…")));
      } else if (r.status !== "active") content = reviewStep(r);
      else if (r.step === "face") content = faceStep(r);
      else content = gestureStep(r);
      fill(root, 
        h("div", { class: "row between" }, stepper(step),
          r.status === "active" ? h("button", { class: "btn sm ghost", type: "button", "aria-keyshortcuts": "Escape",
            onclick: cancel }, icon("x"), "Cancel", kbdHint("Esc")) : null),
        h("div", { class: "row small muted" }, icon("register"),
          h("span", null, "Registering ", h("strong", null, r.username), r.replacing ? " (replacing the existing profile)" : "")),
        content);
      if (r.phase === "recording" && r.recording && cam) cam.setProgress(r.recording);
    };

    unsub = app.subscribe((st) => {
      if (cam) cam.update(st.camera);
      if (suppress) { if (!st.register) suppress = false; return; }
      if (st.register) render(st.register);
    });
    releasePoll = app.needFastPoll();

    let current = null;
    try { current = (await api.status()).register; } catch (e) { /* offline */ }
    if (current) render(current); else showAccount();

    return async () => {
      releaseKeys();
      if (unsub) unsub();
      if (releasePoll) releasePoll();
      dropCam();
      if (reg && reg.status === "active") {
        try { await api.registerCancel(); toast("Registration cancelled. Nothing was saved.", "warn"); } catch (e) { /* ignore */ }
      }
      if (reg) { try { await api.registerClose(); } catch (e) { /* ignore */ } }
    };
  },
};
