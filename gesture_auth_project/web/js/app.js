// WaveLock Console — shell: router, theme, status polling, camera lifecycle.

import { api } from "./api.js";
import { h, icon, clear, toast, fill, add } from "./ui.js";
import dashboard from "./views/dashboard.js";
import signin from "./views/signin.js";
import register from "./views/register.js";
import users from "./views/users.js";
import security from "./views/security.js";
import settings from "./views/settings.js";

const VIEWS = [dashboard, signin, register, users, security, settings];
const FAST_POLL_MS = 200;
const SLOW_POLL_MS = 3000;

class App {
  constructor() {
    this.system = null;
    this.status = null;
    this.listeners = new Set();
    this.view = null;
    this.cleanup = null;
    this.fastPoll = 0;            // number of views/components that need fast polling
    this.pollTimer = null;
    this.navToken = 0;
  }

  get debug() { return Boolean(this.system && this.system.debug); }

  async init() {
    this.buildNav();
    this.initTheme();
    try {
      this.system = await api.system();
    } catch (e) {
      toast(e.message, "danger", 8000);
    }
    this.renderModeBadge();
    window.addEventListener("hashchange", () => this.route());
    await this.route();
    this.poll();
  }

  // ---------------------------------------------------------------- theme
  initTheme() {
    const btn = document.getElementById("theme-toggle");
    const paint = () => {
      const dark = document.documentElement.getAttribute("data-theme") === "dark";
      fill(btn, icon(dark ? "sun" : "moon"));
      btn.setAttribute("role", "switch");
      btn.setAttribute("aria-checked", String(dark));
      btn.setAttribute("aria-label", "Dark mode");
      btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    };
    btn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      this.setTheme(next);
      paint();
    });
    this.paintThemeButton = paint;
    paint();
  }

  setTheme(theme) {
    let value = theme;
    if (theme === "system") {
      value = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
      try { localStorage.removeItem("wavelock-theme"); } catch (e) { /* ignore */ }
    } else {
      try { localStorage.setItem("wavelock-theme", theme); } catch (e) { /* ignore */ }
    }
    document.documentElement.setAttribute("data-theme", value);
    if (this.paintThemeButton) this.paintThemeButton();
  }

  themePreference() {
    try { return localStorage.getItem("wavelock-theme") || "system"; } catch (e) { return "system"; }
  }

  // ---------------------------------------------------------------- nav / routing
  buildNav() {
    const nav = document.getElementById("nav");
    for (const v of VIEWS) {
      nav.append(h("a", { href: `#/${v.id}`, dataset: { view: v.id }, title: v.navLabel },
        icon(v.icon), h("span", null, v.navLabel)));
    }
  }

  renderModeBadge() {
    const root = clear(document.getElementById("mode-badge"));
    if (!this.system) return;
    add(root, this.debug
      ? h("div", { class: "mode-card dev" }, icon("code"),
        h("div", null, h("strong", null, "Developer mode"),
          "Scores and reasons are visible. Not for real sign-ins."))
      : h("div", { class: "mode-card" }, icon("eyeOff"),
        h("div", null, h("strong", null, "Protected mode"),
          "Only the decision is shown, so an observer cannot tune an imitation.")));
  }

  async route() {
    const token = ++this.navToken;
    const [path, query] = (location.hash.replace(/^#\/?/, "") || "dashboard").split("?");
    const view = VIEWS.find((v) => v.id === path) || dashboard;
    const params = new URLSearchParams(query || "");

    if (this.cleanup) {
      const done = this.cleanup;
      this.cleanup = null;
      try { await done(); } catch (e) { /* view cleanup errors are not fatal */ }
    }
    if (token !== this.navToken) return;
    if (this.view && this.view.camera && !view.camera) this.releaseCamera();

    this.view = view;
    document.title = `${view.title} · WaveLock`;
    document.getElementById("page-title").textContent = view.title;
    document.getElementById("page-sub").textContent = view.subtitle;
    for (const a of document.querySelectorAll("#nav a")) {
      if (a.dataset.view === view.id) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
    const root = clear(document.getElementById("view"));
    const container = h("div", { class: "stack" });
    add(root, container);
    try {
      this.cleanup = (await view.mount(container, this, params)) || null;
    } catch (e) {
      container.append(h("div", { class: "card" }, h("p", null, `This page failed to load: ${e.message}`)));
    }
    root.focus({ preventScroll: true });
  }

  go(path) {
    if (location.hash === `#/${path}`) this.route();
    else location.hash = `#/${path}`;
  }

  // ---------------------------------------------------------------- status polling
  subscribe(fn) {
    this.listeners.add(fn);
    if (this.status) fn(this.status);
    return () => this.listeners.delete(fn);
  }

  needFastPoll() {
    this.fastPoll += 1;
    this.poll();
    let released = false;
    return () => { if (!released) { released = true; this.fastPoll = Math.max(0, this.fastPoll - 1); } };
  }

  async poll() {
    clearTimeout(this.pollTimer);
    try {
      this.status = await api.status();
      this.renderCameraPill();
      for (const fn of this.listeners) {
        try { fn(this.status); } catch (e) { console.error(e); }
      }
    } catch (e) {
      this.renderCameraPill(true);
    }
    this.pollTimer = setTimeout(() => this.poll(), this.fastPoll > 0 ? FAST_POLL_MS : SLOW_POLL_MS);
  }

  renderCameraPill(offline = false) {
    const pillEl = document.getElementById("camera-pill");
    const cam = this.status && this.status.camera;
    let cls = "pill", text = "Camera off", dot = "dot";
    if (offline) { cls += " danger"; text = "Server offline"; }
    else if (cam && cam.state === "running") { cls += " ok"; text = "Camera live"; dot += " pulse"; }
    else if (cam && cam.state === "starting") { cls += " warn"; text = "Starting camera"; }
    else if (cam && cam.state === "error") { cls += " danger"; text = "Camera error"; }
    pillEl.className = cls;
    fill(pillEl, h("span", { class: dot }), text);
  }

  // ---------------------------------------------------------------- camera lifecycle
  async ensureCamera() {
    const cam = this.status && this.status.camera;
    if (cam && cam.state === "running") return cam;
    const started = await api.cameraStart(null);
    this.poll();
    return started;
  }

  async releaseCamera() {
    try { await api.cameraStop(); } catch (e) { /* busy or already stopped */ }
    this.poll();
  }
}

const app = new App();
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", () => app.init(), { once: true });
} else {
  app.init();     // module scripts run after parsing, so this is the usual path
}
