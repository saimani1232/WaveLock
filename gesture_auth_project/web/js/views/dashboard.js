import { api } from "../api.js";
import { handMark } from "../brand.js";
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

// Seven-day sign-in chart: stacked bars (granted / denied) per day, built as
// inline SVG from the existing /api/activity log (no chart library).
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  for (const k of kids) el.append(k);
  return el;
}

function dayKey(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function weekChart(entries) {
  const days = [];
  const today = new Date();
  for (let i = 6; i >= 0; i--) {
    const d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - i);
    days.push({ key: dayKey(d), label: d.toLocaleDateString([], { weekday: "short" }),
      full: d.toLocaleDateString([], { weekday: "long", day: "numeric", month: "short" }), granted: 0, denied: 0 });
  }
  const byKey = Object.fromEntries(days.map((d) => [d.key, d]));
  for (const e of entries) {
    const d = byKey[(e.time || "").slice(0, 10)];
    if (d) d[e.granted ? "granted" : "denied"] += 1;
  }
  const total = days.reduce((n, d) => n + d.granted + d.denied, 0);
  const legend = h("div", { class: "chart-legend" },
    h("span", null, h("i", { class: "sw ok" }), "Granted"), h("span", null, h("i", { class: "sw danger" }), "Denied"));
  if (!total) {
    return h("div", { class: "stack" }, legend,
      h("div", { class: "empty" }, icon("activity"), h("p", null, "No sign-in attempts in the last 7 days.")));
  }
  const W = 700, H = 210, top = 22, bottom = 30, left = 8, right = 8;
  const max = Math.max(...days.map((d) => d.granted + d.denied));
  const step = (W - left - right) / days.length, bw = Math.min(44, step * 0.46);
  const y = (v) => top + (H - top - bottom) * (1 - v / max);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "week-chart", role: "img",
    "aria-label": `Sign-in attempts per day for the last 7 days, ${total} in total` });
  svg.append(svgEl("line", { x1: left, x2: W - right, y1: H - bottom, y2: H - bottom, class: "axis" }));
  days.forEach((d, i) => {
    const cx = left + step * (i + 0.5), x = cx - bw / 2;
    const g = svgEl("g", { class: "bar-group", tabindex: "0" },
      svgEl("title", {}, `${d.full}: ${d.granted} granted, ${d.denied} denied`));
    g.append(svgEl("rect", { x: cx - step / 2, y: top, width: step, height: H - top - bottom, class: "hit" }));
    const yG = y(d.granted), yAll = y(d.granted + d.denied);
    if (d.granted) g.append(svgEl("rect", { x, y: yG, width: bw, height: H - bottom - yG, rx: 4, class: "seg ok" }));
    if (d.denied) g.append(svgEl("rect", { x, y: yAll, width: bw, height: Math.max(0, yG - yAll - (d.granted ? 2 : 0)), rx: 4, class: "seg danger" }));
    const n = d.granted + d.denied;
    if (n) g.append(svgEl("text", { x: cx, y: yAll - 7, class: "val" }, String(n)));
    g.append(svgEl("text", { x: cx, y: H - 10, class: "day" }, d.label));
    svg.append(g);
  });
  return h("div", { class: "stack" }, legend, svg);
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
      let users, activity, week;
      try {
        [users, activity, week] = await Promise.all([api.users(), api.activity(8), api.activity(500)]);
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
          h("div", { class: "stack" },
          h("div", { class: "card" },
            h("div", { class: "card-head" },
              h("div", null, h("h2", null, "Sign-ins this week"),
                h("div", { class: "sub" }, "Granted and denied attempts per day"))),
            weekChart(week.entries)),
          h("div", { class: "card" },
            h("div", { class: "card-head" },
              h("div", null, h("h2", null, "Recent activity"),
                h("div", { class: "sub" }, "Latest sign-in attempts from this computer")),
              h("a", { class: "btn sm ghost", href: "#/security" }, "View all", icon("arrowRight"))),
            activityList(activity.entries, app.debug))),

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
            : h("div", { class: "empty hero-empty" }, handMark({ animated: true, className: "empty-hand" }),
              h("h3", null, "No hands enrolled yet"),
              h("p", null, "Register a user: a face anchor and five samples of a personal gesture."),
              h("a", { class: "btn primary sm", href: "#/register" }, icon("register"), "Register the first hand"))));
    };

    await render();
    const timer = setInterval(render, 15000);
    return () => clearInterval(timer);
  },
};
