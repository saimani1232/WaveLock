import { api } from "../api.js";
import { CameraPanel } from "../camera.js";
import { h, icon, clear, toast, modal, notice, initials, pill, fmtTime, fmtNum, fmtSeconds, poseLabel, fill, add } from "../ui.js";

function faceDialog(app, user, onDone) {
  const cam = new CameraPanel({ showFace: true, label: "Live camera for face enrolment" });
  const dots = h("div", { class: "face-dots" });
  const status = h("p", { class: "small muted", "aria-live": "polite" },
    "Look straight at the camera in even light, then start the capture.");
  const startBtn = h("button", { class: "btn primary", type: "button", disabled: true }, icon("face"), "Capture face");
  let unsub = null, releasePoll = null, running = false, finished = false;

  const paintDots = (n, total) => {
    fill(dots, Array.from({ length: total }, (_, i) => h("span", { class: i < n ? "on" : "" })));
  };
  paintDots(0, 5);

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    try {
      await api.faceEnroll(user.name);
      running = true;
      status.textContent = "Capturing… hold still.";
      app.poll();
    } catch (e) {
      startBtn.disabled = false;
      if (e.code === "owner_signin_required") {
        // only the owner may change the face of a profile: offer the sign-in
        startBtn.disabled = true;
        fill(status, h("span", null, e.message, " "),
          h("button", { class: "btn sm primary", type: "button", onclick: () => {
            dlg.close(null);
            app.go(`signin?user=${encodeURIComponent(user.name)}`);
          } }, icon("signin"), `Sign in as ${user.name}`));
      } else {
        toast(e.message, "danger");
      }
    }
  });

  const body = h("div", { class: "stack" },
    user.face_enrolled ? notice("warn", "alert", "This replaces the face currently enrolled for ",
      h("strong", null, user.name), ". Only do this for the profile owner.") : null,
    cam.el, h("div", { class: "row" }, dots, h("span", { class: "spacer" }), startBtn), status);

  const dlg = modal({
    title: `${user.face_enrolled ? "Replace" : "Enrol"} face · ${user.name}`, body, wide: true,
    actions: [{ label: "Close", kind: "ghost", value: null }],
    onClose: async () => {
      if (unsub) unsub();
      if (releasePoll) releasePoll();
      cam.stop();
      try { await api.faceClose(); } catch (e) { /* ignore */ }
      app.releaseCamera();
      if (finished) onDone();
    },
  });
  releasePoll = app.needFastPoll();
  unsub = app.subscribe((st) => {
    cam.update(st.camera);
    if (!running && !finished) startBtn.disabled = !cam.live;   // only with a live picture
    const f = st.face;
    if (!running || !f || f.username !== user.name) return;
    paintDots(f.progress, f.required);
    if (f.status === "done") {
      running = false;
      finished = true;
      status.textContent = f.message;
      cam.setMode("granted", { title: "Face enrolled", sub: "Checked at every sign-in" });
      toast(`Face enrolled for ${user.name}.`, "ok");
    } else if (f.status === "failed") {
      running = false;
      startBtn.disabled = false;
      status.textContent = f.message;
      cam.setMode("denied", { title: "Capture failed", sub: "Try again in better light" });
    }
  });
  cam.setCoach({ needFace: true, needHand: false, text: "Look straight at the camera" });
  cam.start(app);
  return dlg;
}

function userCard(u, app, refresh) {
  if (u.error) {
    return h("div", { class: "card user-card" },
      h("div", { class: "user-head" }, h("span", { class: "avatar" }, initials(u.name)),
        h("div", { class: "name" }, u.name)),
      notice("danger", "alert", u.error),
      h("div", { class: "actions" },
        h("a", { class: "btn sm", href: `#/register?user=${encodeURIComponent(u.name)}` }, icon("refresh"), "Re-register")));
  }
  const t = u.thresholds;
  return h("div", { class: "card user-card" },
    h("div", { class: "user-head" },
      h("span", { class: "avatar lg" }, initials(u.name)),
      h("div", { class: "spacer" }, h("div", { class: "name" }, u.name),
        h("div", { class: "small muted" }, u.last_attempt
          ? `Last attempt ${fmtTime(u.last_attempt.time)} · ${u.last_attempt.granted ? "granted" : "denied"}`
          : "No sign-in attempts yet"))),
    h("div", { class: "chips" },
      u.face_enrolled ? pill("Face anchor", "accent", "face") : pill("No face anchor", "warn", "face"),
      pill(`${u.samples} samples`, "", "wave"),
      pill(`${poseLabel(u.pose_method)} pose`, ""),
      u.path_gate ? pill("Wrist-path gate", "info") : pill("No path gate (legacy)", ""),
      u.locked_for > 0 ? pill(`Locked ${fmtSeconds(u.locked_for)}`, "danger", "lock") : null),
    t ? h("div", { class: "thresholds" },
      [["DTW", t.dtw], ["Finger", t.finger_state], ["Order", t.transition], ["Segment", t.segment],
        ["Path", t.trajectory], ["Anatomy", t.anthro_tolerance]]
        .map(([k, v]) => h("div", null, h("span", null, k), h("b", null, v === null || v === undefined ? "off" : fmtNum(v, 3)))))
      : null,
    h("div", { class: "actions" },
      h("a", { class: "btn sm primary", href: `#/signin?user=${encodeURIComponent(u.name)}` }, icon("signin"), "Sign in"),
      h("button", { class: "btn sm", type: "button", onclick: () => faceDialog(app, u, refresh) },
        icon("face"), u.face_enrolled ? "Replace face" : "Enrol face"),
      h("a", { class: "btn sm ghost", href: `#/register?user=${encodeURIComponent(u.name)}` }, icon("refresh"), "Re-register")));
}

export default {
  id: "users",
  navLabel: "Users",
  title: "Users",
  subtitle: "Registered profiles, their protections and quick actions.",
  icon: "users",
  camera: false,

  async mount(root, app) {
    let all = [];
    const search = h("input", { class: "input", type: "search", placeholder: "Filter by name", "aria-label": "Filter users" });
    const grid = h("div", { class: "grid grid-3" });
    const draw = () => {
      const q = search.value.trim().toLowerCase();
      const list = all.filter((u) => u.name.includes(q));
      clear(grid);
      if (!all.length) {
        add(grid, h("div", { class: "card empty" }, icon("users"), h("p", null, "No users registered yet."),
          h("a", { class: "btn primary sm", href: "#/register" }, icon("register"), "Register a user")));
      } else if (!list.length) {
        add(grid, h("div", { class: "card empty" }, icon("users"), h("p", null, "No user matches this filter.")));
      } else {
        add(grid, list.map((u) => userCard(u, app, refresh)));
      }
    };
    const refresh = async () => {
      try { all = await api.users(); } catch (e) { toast(e.message, "danger"); }
      draw();
    };
    search.addEventListener("input", draw);
    add(root, 
      h("div", { class: "row between" },
        h("div", { class: "row" }, h("div", { class: "field" }, search)),
        h("div", { class: "row" },
          h("button", { class: "btn", type: "button", onclick: refresh }, icon("refresh"), "Refresh"),
          h("a", { class: "btn primary", href: "#/register" }, icon("register"), "New user"))),
      app.debug ? notice("warn", "code", "Developer mode: calibrated thresholds are shown on each card.") : null,
      grid);
    await refresh();
    const timer = setInterval(refresh, 20000);
    return () => clearInterval(timer);
  },
};
