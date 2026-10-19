// DOM helpers, icons, toasts and modals. User data is always inserted as
// text nodes (never innerHTML); only the static icon strings below are
// parsed as markup.

const ICON_PATHS = {
  dashboard: '<rect x="3" y="3" width="7" height="9" rx="2"/><rect x="14" y="3" width="7" height="5" rx="2"/><rect x="14" y="12" width="7" height="9" rx="2"/><rect x="3" y="16" width="7" height="5" rx="2"/>',
  signin: '<path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M11 11.5V4a1.5 1.5 0 0 1 3 0v8"/><path d="M14 11V5.5a1.5 1.5 0 0 1 3 0V13"/><path d="M17 8.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1.2a7 7 0 0 1-5.4-2.6L3.6 16a1.6 1.6 0 0 1 2.4-2.1L8 16"/>',
  register: '<circle cx="9" cy="8" r="4"/><path d="M2.5 21a6.5 6.5 0 0 1 13 0"/><path d="M19 8v6M16 11h6"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7"/><path d="M18.5 14.5A6.5 6.5 0 0 1 21.5 20"/>',
  security: '<path d="M12 2.5 4.5 5.5v6c0 4.6 3.1 8.6 7.5 10 4.4-1.4 7.5-5.4 7.5-10v-6z"/><path d="m8.8 12 2.2 2.2 4.4-4.6"/>',
  settings: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5z"/>',
  camera: '<path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.8l1.4-2h6.6l1.4 2h1.8A2.5 2.5 0 0 1 21 8.5v9a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z"/><circle cx="12" cy="13" r="3.8"/>',
  cameraOff: '<path d="m3 3 18 18"/><path d="M8.7 4h6.6l1.4 2h1.8A2.5 2.5 0 0 1 21 8.5v8.8M18.5 20h-13A2.5 2.5 0 0 1 3 17.5v-9A2.5 2.5 0 0 1 5.3 6"/><path d="M9.4 10.4a3.8 3.8 0 0 0 5.2 5.2"/>',
  hand: '<path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M11 11.5V4a1.5 1.5 0 0 1 3 0v8"/><path d="M14 11V5.5a1.5 1.5 0 0 1 3 0V13"/><path d="M17 8.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1.2a7 7 0 0 1-5.4-2.6L3.6 16a1.6 1.6 0 0 1 2.4-2.1L8 16"/>',
  face: '<circle cx="12" cy="12" r="9.5"/><path d="M8.5 14.5a4.5 4.5 0 0 0 7 0"/><path d="M9 9.5h.01M15 9.5h.01"/>',
  scan: '<path d="M3 8V5.5A2.5 2.5 0 0 1 5.5 3H8M16 3h2.5A2.5 2.5 0 0 1 21 5.5V8M21 16v2.5a2.5 2.5 0 0 1-2.5 2.5H16M8 21H5.5A2.5 2.5 0 0 1 3 18.5V16"/><path d="M7 12h10"/>',
  bones: '<path d="M7 21v-6M12 21V11M17 21v-8"/><path d="M5 5.5h4M10 3.5h4M15 6.5h4"/><circle cx="7" cy="12" r="2"/><circle cx="12" cy="8" r="2"/><circle cx="17" cy="10" r="2"/>',
  decision: '<circle cx="12" cy="12" r="9.5"/><path d="m8 12.3 2.8 2.7L16.3 9"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  alert: '<path d="M12 3.5 2.5 20h19z"/><path d="M12 10v4.5M12 17.5h.01"/>',
  info: '<circle cx="12" cy="12" r="9.5"/><path d="M12 11v5.5M12 7.5h.01"/>',
  lock: '<rect x="4.5" y="10.5" width="15" height="10.5" rx="2.5"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
  unlock: '<rect x="4.5" y="10.5" width="15" height="10.5" rx="2.5"/><path d="M8 10.5V7.5a4 4 0 0 1 7.6-1.7"/>',
  clock: '<circle cx="12" cy="12" r="9.5"/><path d="M12 7v5.5l3.5 2"/>',
  record: '<circle cx="12" cy="12" r="9.5"/><circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/>',
  refresh: '<path d="M20.5 12A8.5 8.5 0 1 1 18 6"/><path d="M20.5 3.5V8H16"/>',
  arrowRight: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  logout: '<path d="M14.5 4H18a2.5 2.5 0 0 1 2.5 2.5v11A2.5 2.5 0 0 1 18 20h-3.5"/><path d="M10 16.5 5.5 12 10 7.5M5.5 12H16"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  code: '<path d="m8 7-5 5 5 5M16 7l5 5-5 5"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9.5 9.5h5v5h-5zM9 2.5V6M15 2.5V6M9 18v3.5M15 18v3.5M2.5 9H6M2.5 15H6M18 9h3.5M18 15h3.5"/>',
  flask: '<path d="M9 3h6M10 3v6.5L4.8 18.6A1.6 1.6 0 0 0 6.2 21h11.6a1.6 1.6 0 0 0 1.4-2.4L14 9.5V3"/><path d="M7.5 15h9"/>',
  monitor: '<rect x="2.5" y="4" width="19" height="13" rx="2.5"/><path d="M8 21h8M12 17v4"/>',
  wave: '<path d="M2.5 12c2.4-3.5 4.8-3.5 7.1 0s4.8 3.5 7.2 0 3.2-2.9 4.7-1.6"/>',
  eyeOff: '<path d="m3 3 18 18"/><path d="M10.6 5.2A9.9 9.9 0 0 1 12 5c5 0 8.5 4.5 9.5 7a13 13 0 0 1-2.4 3.5M6.6 6.6A13.4 13.4 0 0 0 2.5 12c1 2.5 4.5 7 9.5 7a9.6 9.6 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>',
  trash: '<path d="M4 7h16M9.5 7V4.5h5V7M6.5 7l1 13.5h9L17.5 7"/>',
};

export function icon(name, extraClass = "") {
  const span = document.createElement("span");
  span.className = `ic ${extraClass}`.trim();
  span.setAttribute("aria-hidden", "true");
  // width/height are only defaults: any CSS width/height rule overrides them.
  span.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" ` +
    `stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[name] || ""}</svg>`;
  return span.firstChild;   // the <svg> element itself
}

// h("div", {class: "card", onclick: fn}, child, "text", [children...])
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "vars") for (const [n, val] of Object.entries(v)) el.style.setProperty(n, val);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

// Like el.append(), but flattens arrays and skips null/false (native append
// would print "null" or "[object HTMLDivElement]").
export function add(el, ...children) {
  append(el, children);
  return el;
}

export function fill(el, ...children) {
  return add(clear(el), ...children);
}

export function initials(name) {
  const parts = String(name || "?").split(/[_.\-\s]+/).filter(Boolean);
  return ((parts[0] || "?")[0] + (parts[1] ? parts[1][0] : "")).toUpperCase();
}

export function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return sameDay ? `Today ${time}` : `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

export function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

export function fmtSeconds(s) {
  s = Math.max(0, Math.ceil(s || 0));
  if (s < 60) return `${s} s`;
  if (s % 60 === 0) return `${s / 60} min`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

export function poseLabel(method) {
  return method === "aspect_x_wrist_unit_v2" ? "Aspect-corrected" : "Standard";
}

// ------------------------------------------------------------------ toasts
export function toast(message, kind = "info", ms = 4200) {
  const root = document.getElementById("toasts");
  const names = { ok: "check", danger: "alert", warn: "alert", info: "info" };
  const t = h("div", { class: `toast ${kind}`, role: kind === "danger" ? "alert" : "status" },
    icon(names[kind] || "info"), h("div", null, message));
  root.appendChild(t);
  setTimeout(() => {
    t.classList.add("leave");
    setTimeout(() => t.remove(), 260);
  }, ms);
}

// ------------------------------------------------------------------ modal
export function modal({ title, body, actions = [], wide = false, onClose = null }) {
  const root = document.getElementById("modal-root");
  const previous = document.activeElement;
  let closed = false;
  const close = (value) => {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKey);
    backdrop.remove();
    if (onClose) onClose(value);
    if (previous && previous.focus) previous.focus();
  };
  const onKey = (e) => { if (e.key === "Escape") close(null); };
  const titleId = `m-${Math.random().toString(36).slice(2)}`;
  const box = h("div", { class: `modal ${wide ? "wide" : ""}`, role: "dialog", "aria-modal": "true", "aria-labelledby": titleId },
    h("h2", { id: titleId, text: title }), body,
    actions.length ? h("div", { class: "modal-actions" },
      actions.map((a) => h("button", {
        class: `btn ${a.kind || ""}`, type: "button",
        onclick: () => (a.onClick ? a.onClick(close) : close(a.value)),
      }, a.icon ? icon(a.icon) : null, a.label))) : null);
  const backdrop = h("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === backdrop) close(null); } }, box);
  root.appendChild(backdrop);
  document.addEventListener("keydown", onKey);
  const focusable = box.querySelector("button.primary, button, input");
  if (focusable) setTimeout(() => focusable.focus(), 30);
  return { close, box };
}

export function confirmDialog({ title, message, confirmLabel = "Continue", danger = false }) {
  return new Promise((resolve) => {
    modal({
      title,
      body: h("p", { class: "muted", text: message }),
      actions: [
        { label: "Cancel", kind: "ghost", value: false },
        { label: confirmLabel, kind: danger ? "danger" : "primary", value: true },
      ],
      onClose: (v) => resolve(Boolean(v)),
    });
  });
}

export function notice(kind, iconName, ...content) {
  return h("div", { class: `notice ${kind}` }, icon(iconName), h("div", null, ...content));
}

export function pill(text, kind = "", iconName = null, dot = false) {
  return h("span", { class: `pill sm ${kind}` }, dot ? h("span", { class: "dot" }) : null,
    iconName ? icon(iconName) : null, text);
}

// ------------------------------------------------------------------ keyboard shortcuts
// One page at a time registers a provider that returns the shortcuts that
// apply *right now* (e.g. T only after a result). Every page follows the same
// safety rules, so a key never fires by accident:
//   - not while typing in a text box, list or editable element,
//   - not with Ctrl / Alt / Cmd held (browser shortcuts keep working),
//   - not while a dialog is open (Esc belongs to the dialog),
//   - not on key repeat (holding R records once),
//   - Space is left to a focused button, which presses it natively.
// Bindings: [{ keys: ["t"], label: "Try again", run: () => ... }]
let activeProvider = null;

function typingTarget(el) {
  return !!(el && el.closest && el.closest("input, textarea, select, [contenteditable=''], [contenteditable='true']"));
}

function keyName(e) {
  if (e.key === " " || e.key === "Spacebar") return "space";
  if (e.key === "Escape" || e.key === "Esc") return "escape";
  return e.key.length === 1 ? e.key.toLowerCase() : e.key.toLowerCase();
}

export function useShortcuts(provider) {
  activeProvider = provider;
  const onKey = (e) => {
    if (e.defaultPrevented || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    if (typingTarget(e.target) || document.querySelector(".modal-backdrop")) return;
    const name = keyName(e);
    if (name === "space" && e.target.closest && e.target.closest("button, a")) return;
    const binding = (provider() || []).find((b) => b.keys.includes(name));
    if (!binding) return;
    e.preventDefault();
    binding.run();
  };
  document.addEventListener("keydown", onKey);
  return () => {
    document.removeEventListener("keydown", onKey);
    if (activeProvider === provider) activeProvider = null;
  };
}

export function activeShortcuts() {
  try { return activeProvider ? activeProvider() || [] : []; } catch (e) { return []; }
}

// Small key badge shown inside a button; the button also announces the key.
export function kbdHint(key) {
  return h("span", { class: "kbd-hint", "aria-hidden": "true" }, key);
}

const KEY_LABEL = { space: "Space", escape: "Esc" };
export function keyLabel(k) { return KEY_LABEL[k] || k.toUpperCase(); }
