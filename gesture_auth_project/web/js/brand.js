// WaveLock brand mark: an open hand drawn from MediaPipe's 21 hand landmarks,
// with a keyhole in the palm ("your hand is your password").
// Static markup only (no user data), so building it with innerHTML is safe.
// No style="" attributes: the page's Content-Security-Policy (style-src 'self')
// blocks them, so animation delays are passed as data-* attributes and set
// as CSS variables from script (CSSOM changes are allowed by the policy).

// Landmark positions in a 64 x 64 box (wrist = 0, thumb 1-4, index 5-8,
// middle 9-12, ring 13-16, pinky 17-20), same order as MediaPipe Hands.
const P = [
  [32, 57], [24.5, 51.5], [18.5, 45.5], [14.5, 39.5], [11.5, 34],
  [25, 35], [23.2, 25], [22.2, 18.6], [21.4, 12.6],
  [32, 33], [32, 22], [32, 15.4], [32, 9.2],
  [39, 35], [40.8, 25.4], [41.8, 19.4], [42.6, 14],
  [45, 39], [48, 32], [49.8, 27.2], [51.2, 22.6],
];
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
// Fingertips light up in the order of an example gesture: index, middle,
// pinky, ring (the "1-2-4-3" sequence the enrolment tips suggest).
const TIP_ORDER = { 8: 0, 12: 1, 20: 2, 16: 3 };
const KEYHOLE = "M32 39.6a3.4 3.4 0 0 1 1.7 6.34L35 51h-6l1.3-5.06A3.4 3.4 0 0 1 32 39.6z";

let uid = 0;

export function handMarkSVG({ animated = false, tile = true } = {}) {
  const id = `wl${++uid}`;
  const bones = BONES.map(([a, b], i) =>
    `<line class="hm-bone" data-i="${i}" x1="${P[a][0]}" y1="${P[a][1]}" x2="${P[b][0]}" y2="${P[b][1]}"/>`).join("");
  const joints = P.map(([x, y], i) => {
    const tip = TIP_ORDER[i];
    return `<circle class="hm-joint${tip !== undefined ? " hm-tip" : ""}"${tip !== undefined ? ` data-t="${tip}"` : ""} ` +
      `cx="${x}" cy="${y}" r="${tip !== undefined ? 2.3 : 1.7}"/>`;
  }).join("");
  return `<svg class="hand-mark${animated ? " animated" : ""}" viewBox="0 0 64 64" role="img" aria-label="WaveLock">` +
    `<defs><linearGradient id="${id}g" x1="10" y1="6" x2="54" y2="60" gradientUnits="userSpaceOnUse">` +
    `<stop offset="0" stop-color="#99f6e4"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>` +
    (tile ? `<rect class="hm-tile" x="1" y="1" width="62" height="62" rx="16"/>` : "") +
    (animated ? `<circle class="hm-orbit" cx="32" cy="33" r="28.5" stroke="url(#${id}g)"/>` : "") +
    `<g stroke="url(#${id}g)" fill="none" stroke-width="2" stroke-linecap="round">${bones}</g>` +
    `<g fill="url(#${id}g)">${joints}</g>` +
    `<path class="hm-keyhole" d="${KEYHOLE}"/></svg>`;
}

export function handMark(opts) {
  const wrap = document.createElement("span");
  wrap.className = `hand-mark-wrap ${opts && opts.className ? opts.className : ""}`.trim();
  wrap.innerHTML = handMarkSVG(opts);
  for (const el of wrap.querySelectorAll("[data-i]")) el.style.setProperty("--i", el.dataset.i);
  for (const el of wrap.querySelectorAll("[data-t]")) el.style.setProperty("--t", el.dataset.t);
  return wrap;
}
