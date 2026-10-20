// Face pipeline of the desktop app (OpenCV 4.11 FaceDetectorYN + FaceRecognizerSF,
// utils/face_auth.py), in plain JavaScript for the mobile app. No DOM: images
// are {data: RGBA Uint8ClampedArray, width, height}. The models run in ONNX
// Runtime Web (face_worker.js); tests/face_parity.mjs compares every step
// with OpenCV on the same image.

export const YUNET_SIZE = 640;           // the ONNX file's fixed input (640 x 640)
export const YUNET_SCORE = 0.70;         // utils/face_auth.get_face_detector
export const YUNET_NMS = 0.30;
export const YUNET_TOP_K = 5000;
export const PRIMARY_MIN_SCORE = 0.70;   // detect_primary_face(min_confidence)
const STRIDES = [8, 16, 32];

// YuNet input: BGR float32, 0..255, NCHW. OpenCV pads the frame with zeros on
// the right / bottom to a multiple of 32; the ONNX file needs 640 x 640, so
// the padding goes up to 640 (frames are at most 640 pixels on each side).
export function yunetInput(img) {
  const { data, width, height } = img;
  if (width > YUNET_SIZE || height > YUNET_SIZE) throw new Error("frame larger than 640 px");
  const plane = YUNET_SIZE * YUNET_SIZE;
  const out = new Float32Array(3 * plane);           // zeros = padding
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = (y * width + x) * 4, d = y * YUNET_SIZE + x;
      out[d] = data[s + 2];                          // B
      out[plane + d] = data[s + 1];                  // G
      out[2 * plane + d] = data[s];                  // R
    }
  }
  return out;
}

// Decode the 12 YuNet outputs (cls / obj / bbox / kps for strides 8, 16, 32)
// exactly like FaceDetectorYNImpl::postProcess, then NMS. Each face row:
// [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y, score].
export function yunetDecode(outputs, size = YUNET_SIZE) {
  const faces = [];
  STRIDES.forEach((s) => {
    const cols = Math.floor(size / s), rows = Math.floor(size / s);
    const cls = outputs[`cls_${s}`], obj = outputs[`obj_${s}`];
    const bbox = outputs[`bbox_${s}`], kps = outputs[`kps_${s}`];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = Math.min(1, Math.max(0, cls[idx]));
        const objScore = Math.min(1, Math.max(0, obj[idx]));
        const score = Math.fround(Math.sqrt(clsScore * objScore));
        if (score < YUNET_SCORE) continue;
        const cx = (c + bbox[idx * 4]) * s, cy = (r + bbox[idx * 4 + 1]) * s;
        const w = Math.exp(bbox[idx * 4 + 2]) * s, h = Math.exp(bbox[idx * 4 + 3]) * s;
        const row = [cx - w / 2, cy - h / 2, w, h];
        for (let n = 0; n < 5; n++) {
          row.push((kps[idx * 10 + 2 * n] + c) * s, (kps[idx * 10 + 2 * n + 1] + r) * s);
        }
        row.push(score);
        faces.push(row.map(Math.fround));
      }
    }
  });
  return nms(faces);
}

// cv::dnn::NMSBoxes on integer rectangles (Rect2i(int(x), int(y), int(w), int(h))).
function nms(faces) {
  if (faces.length <= 1) return faces;
  const rect = (f) => [Math.trunc(f[0]), Math.trunc(f[1]), Math.trunc(f[2]), Math.trunc(f[3])];
  const order = faces.map((f, i) => i).sort((a, b) => faces[b][14] - faces[a][14]).slice(0, YUNET_TOP_K);
  const keep = [];
  for (const i of order) {
    const a = rect(faces[i]);
    let ok = true;
    for (const j of keep) {
      const b = rect(faces[j]);
      const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
      const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
      const inter = ix * iy, union = a[2] * a[3] + b[2] * b[3] - inter;
      if (union > 0 && inter / union > YUNET_NMS) { ok = false; break; }
    }
    if (ok) keep.push(i);
  }
  return keep.map((i) => faces[i]);
}

// detect_primary_face: the largest face with score >= 0.70 (or null).
export function primaryFace(faces) {
  let best = null;
  for (const f of faces) {
    if (f[14] < PRIMARY_MIN_SCORE) continue;
    if (!best || f[2] * f[3] > best[2] * best[3]) best = f;
  }
  return best;
}

// FaceRecognizerSF::alignCrop: similarity transform of the 5 landmarks onto
// the 112 x 112 SFace template (Umeyama, as getSimilarityTransformMatrix; for
// 2-D the rotation angle is atan2(A10 - A01, A00 + A11) and the scale
// 2 * sqrt(E^2 + H^2) / var, also when det(A) < 0), then a bilinear warp.
const DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
const DST_MEAN = [56.0262, 71.9008];

export function similarityMatrix(row) {
  const src = [0, 1, 2, 3, 4].map((n) => [Math.fround(row[4 + 2 * n]), Math.fround(row[5 + 2 * n])]);
  const mean = [src.reduce((s, p) => s + p[0], 0) / 5, src.reduce((s, p) => s + p[1], 0) / 5].map(Math.fround);
  const sd = src.map((p) => [Math.fround(p[0] - mean[0]), Math.fround(p[1] - mean[1])]);
  const dd = DST.map((p) => [Math.fround(p[0] - DST_MEAN[0]), Math.fround(p[1] - DST_MEAN[1])]);
  let a00 = 0, a01 = 0, a10 = 0, a11 = 0, v1 = 0, v2 = 0;
  for (let i = 0; i < 5; i++) {
    a00 += dd[i][0] * sd[i][0]; a01 += dd[i][0] * sd[i][1];
    a10 += dd[i][1] * sd[i][0]; a11 += dd[i][1] * sd[i][1];
    v1 += sd[i][0] * sd[i][0]; v2 += sd[i][1] * sd[i][1];
  }
  a00 /= 5; a01 /= 5; a10 /= 5; a11 /= 5; v1 /= 5; v2 /= 5;
  const e = (a00 + a11) / 2, hh = (a10 - a01) / 2;
  const theta = Math.atan2(hh, e);
  const scale = (2 * Math.hypot(e, hh)) / (v1 + v2);
  const c = Math.cos(theta), s = Math.sin(theta);
  const t00 = c * scale, t01 = -s * scale, t10 = s * scale, t11 = c * scale;
  return [t00, t01, DST_MEAN[0] - (t00 * mean[0] + t01 * mean[1]),
    t10, t11, DST_MEAN[1] - (t10 * mean[0] + t11 * mean[1])];
}

// warpAffine(..., INTER_LINEAR, BORDER_CONSTANT 0): every output pixel samples
// the source at M^-1 (x, y). Returns RGB rows (Uint8ClampedArray 112*112*3).
export function warpAffine(img, m, size = 112) {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  const ic = -(ia * c + ib * f), iff = -(id * c + ie * f);
  const { data, width, height } = img;
  const out = new Uint8ClampedArray(size * size * 3);
  const px = (x, y, ch) => (x < 0 || y < 0 || x >= width || y >= height) ? 0 : data[(y * width + x) * 4 + ch];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = ia * x + ib * y + ic, sy = id * x + ie * y + iff;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      for (let ch = 0; ch < 3; ch++) {
        const v = (1 - fy) * ((1 - fx) * px(x0, y0, ch) + fx * px(x0 + 1, y0, ch))
          + fy * ((1 - fx) * px(x0, y0 + 1, ch) + fx * px(x0 + 1, y0 + 1, ch));
        out[(y * size + x) * 3 + ch] = Math.round(v);
      }
    }
  }
  return out;
}

// FaceRecognizerSF::feature: blobFromImage(aligned, 1, 112x112, 0, swapRB=true):
// RGB order, 0..255, NCHW.
export function sfaceInput(rgb, size = 112) {
  const plane = size * size, out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = rgb[i * 3]; out[plane + i] = rgb[i * 3 + 1]; out[2 * plane + i] = rgb[i * 3 + 2];
  }
  return out;
}

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / Math.sqrt(na * nb);
}
