import { api } from "../api.js";
import { h, icon, clear, fmtTime, initials, pill, poseLabel, fmtSeconds, fill, add } from "../ui.js";

function kpi(iconName, value, label, kind = "") {
  return h("div", { class: `card kpi ${kind}` },
    h("div", { class: "kpi-icon" }, icon(iconName)),
    h("div", { class: "kpi-value" }, String(value)),
    h("div", { class: "kpi-label" }, label));
}

function quick(href, iconName, title, sub) {
  return h("a", { class: "quick", href },
    h("span", { class: "q-icon" }, icon(iconName)),
    h("span", null, h("h3", null, title), h("span", { class: "muted small" }, sub)),
    h("span", { class: "q-arrow" }, icon("arrowRight")));
}

export function activityList(entries, debug, emptyText = "No sign-in attempts yet.") {
  if (!entries.length) {
    return h("div", { class: "empty" }, icon("activity"), h("p", null, emptyText));
  }
  return h("div", { class: "list" }, entries.map((e) =>
    h("div", { class: "list-item" },
      h("span", { class: `status-icon ${e.granted ? "ok" : "danger"}` }, icon(e.granted ? "unlock" : "lock")),
      h("div", { class: "spacer" },
        h("div", null, h("strong", null, e.user), " ",
          h("span", { class: "muted" }, e.granted ? "signed in" : "was denied")),
        debug && e.reason ? h("div", { class: "tiny faint mono" }, e.reason) : null),
      h("span", { class: "small muted" }, fmtTime(e.time)))));
}

export default {
  id: "dashboard",
  navLabel: "Dashboard",
  title: "Security overview",
  subtitle: "Profiles, sign-in activity and system health at a glance.",
  icon: "dashboard",
  camera: false,

  async mount(root, app) {
    const body = h("div", { class: "stack" });
    add(root, body);

    const render = async () => {
      let users, activity;
      try {
        [users, activity] = await Promise.all([api.users(), api.activity(8)]);
      } catch (e) {
        fill(body, h("div", { class: "card" }, h("p", null, e.message)));
        return;
      }
      const faces = users.filter((u) => u.face_enrolled).length;
      const sys = app.system || {};
      const cam = (app.status && app.status.camera) || {};
      fill(body, 
        h("div", { class: "grid grid-4" },
          kpi("users", users.length, "Registered users"),
          kpi("face", `${faces}/${users.length}`, "Profiles with face anchor", "info"),
          kpi("activity", activity.today.total,
            `Attempts today · ${activity.today.granted} granted, ${activity.today.denied} denied`, "ok"),
          kpi("lock", activity.locked.length, "Accounts locked right now",
            activity.locked.length ? "danger" : "")),

        h("div", { class: "grid grid-2" },
          quick("#/signin", "signin", "Sign in", "Face + hand gesture, checked together"),
          quick("#/register", "register", "Register a user", "Face anchor and five gesture samples")),

        h("div", { class: "split" },
          h("div", { class: "card" },
            h("div", { class: "card-head" },
              h("div", null, h("h2", null, "Recent activity"),
                h("div", { class: "sub" }, "Latest sign-in attempts from this computer")),
              h("a", { class: "btn sm ghost", href: "#/security" }, "View all", icon("arrowRight"))),
            activityList(activity.entries, app.debug)),

          h("div", { class: "stack" },
            h("div", { class: "card" },
              h("div", { class: "card-head" }, h("h2", null, "System status")),
              h("dl", { class: "kv" },
                h("dt", null, "Camera"), h("dd", null, cam.state === "running" ? "Live" :
                  cam.state === "error" ? "Error" : "Idle (starts on demand)"),
                h("dt", null, "Face models"), h("dd", null,
                  sys.models && sys.models.yunet && sys.models.sface ? "YuNet + SFace ready" : "Missing"),
                h("dt", null, "Mode"), h("dd", null, app.debug ? "Developer" : "Protected"),
                h("dt", null, "Attempt limit"), h("dd", null, sys.limits
                  ? `${sys.limits.max_failures} fails → ${fmtSeconds(sys.limits.lockout_base_sec)} lock` : "—"),
                h("dt", null, "Recording"), h("dd", null, sys.recording_sec ? `${sys.recording_sec} s per attempt` : "—"))),
            activity.locked.length ? h("div", { class: "card" },
              h("div", { class: "card-head" }, h("h2", null, "Locked accounts")),
              h("div", { class: "list" }, activity.locked.map((l) => h("div", { class: "list-item" },
                h("span", { class: "status-icon danger" }, icon("lock")),
                h("strong", { class: "spacer" }, l.user),
                h("span", { class: "small muted" }, `${fmtSeconds(l.locked_for)} left`))))) : null)),

        h("div", { class: "card" },
          h("div", { class: "card-head" },
            h("div", null, h("h2", null, "Profiles"), h("div", { class: "sub" }, "Everyone registered on this computer")),
            h("a", { class: "btn sm ghost", href: "#/users" }, "Manage", icon("arrowRight"))),
          users.length ? h("div", { class: "list" }, users.slice(0, 6).map((u) => h("div", { class: "list-item" },
            h("span", { class: "avatar" }, initials(u.name)),
            h("div", { class: "spacer" }, h("strong", null, u.name),
              h("div", { class: "tiny muted" }, u.error ? u.error :
                `${u.samples} samples · ${poseLabel(u.pose_method)} pose`)),
            u.face_enrolled ? pill("Face", "accent", "face") : pill("No face", "warn"),
            u.locked_for > 0 ? pill("Locked", "danger", "lock") : null)))
            : h("div", { class: "empty" }, icon("users"), h("p", null, "No users yet."),
              h("a", { class: "btn primary sm", href: "#/register" }, icon("register"), "Register the first user"))));
    };

    await render();
    const timer = setInterval(render, 15000);
    return () => clearInterval(timer);
  },
};
