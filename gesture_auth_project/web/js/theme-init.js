// Runs before first paint (blocking, tiny) so the page never flashes the wrong theme.
(function () {
  var pref = null;
  try { pref = localStorage.getItem("wavelock-theme"); } catch (e) { /* storage blocked */ }
  if (pref !== "light" && pref !== "dark") {
    pref = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", pref);
})();
