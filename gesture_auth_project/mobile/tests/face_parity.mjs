// Compares the mobile face pipeline (src/js/local/face_core.js + ONNX
// Runtime Web, the app's own files) with the desktop one (OpenCV 4.11,
// utils/face_auth.py) on a public-domain test image (NASA astronaut portrait
// from scikit-image v0.24.0, downloaded into .cache/testdata, not committed).
//
//   node tests/face_parity.mjs .cache/testdata/astronaut.png

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const mobile = path.dirname(here);
const models = path.join(path.dirname(mobile), "models");
const require = createRequire(import.meta.url);
const ort = require(path.join(mobile, "node_modules", "onnxruntime-web", "dist", "ort.wasm.min.js"));
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";
const fc = await import(pathToFileURL(path.join(mobile, "src", "js", "local", "face_core.js")).href);

const image = process.argv[2];
if (!image) throw new Error("usage: node tests/face_parity.mjs <image.png>");
const work = fs.mkdtempSync(path.join(os.tmpdir(), "wl_face_"));
const failures = [];
const check = (cond, label) => { console.log((cond ? "PASS  " : "FAIL  ") + label); if (!cond) failures.push(label); };

try {
  execFileSync(process.env.PYTHON || "python", [path.join(here, "face_reference.py"), image, work], { stdio: "pipe" });
  const ref = JSON.parse(fs.readFileSync(path.join(work, "reference.json"), "utf8"));
  const yunet = await ort.InferenceSession.create(fs.readFileSync(path.join(models, "face_detection_yunet_2023mar.onnx")));
  const sface = await ort.InferenceSession.create(fs.readFileSync(path.join(models, "face_recognition_sface_2021dec.onnx")));
  const embed = async (rgb) => {
    const out = await sface.run({ data: new ort.Tensor("float32", fc.sfaceInput(rgb), [1, 3, 112, 112]) });
    return Array.from(out.fc1.data);
  };

  for (const [name, r] of Object.entries(ref)) {
    const data = new Uint8ClampedArray(fs.readFileSync(path.join(work, `${name}.rgba`)));
    const img = { data, width: r.width, height: r.height };
    const outs = await yunet.run({ input: new ort.Tensor("float32", fc.yunetInput(img), [1, 3, 640, 640]) });
    const plain = Object.fromEntries(Object.entries(outs).map(([k, v]) => [k, v.data]));
    const faces = fc.yunetDecode(plain);
    const primary = fc.primaryFace(faces);
    check(faces.length === r.faces.length, `${name}: ${faces.length} detection(s), OpenCV ${r.faces.length}`);
    if (!primary || !r.primary) { check(!primary && !r.primary, `${name}: primary face present on both sides`); continue; }
    const rowDiff = Math.max(...primary.slice(0, 14).map((v, i) => Math.abs(v - r.primary[i])));
    check(rowDiff < 1.0, `${name}: primary face box / landmarks within ${rowDiff.toFixed(3)} px of OpenCV`);
    check(Math.abs(primary[14] - r.primary[14]) < 0.01, `${name}: score ${primary[14].toFixed(4)} vs ${r.primary[14].toFixed(4)}`);

    // alignment on OpenCV's row (isolates the warp from the detection)
    const aligned = fc.warpAffine(img, fc.similarityMatrix(r.primary));
    let sum = 0, max = 0;
    for (let i = 0; i < aligned.length; i++) { const d = Math.abs(aligned[i] - r.aligned_rgb[i]); sum += d; max = Math.max(max, d); }
    check(sum / aligned.length < 1.0, `${name}: aligned crop mean |diff| ${(sum / aligned.length).toFixed(3)} (max ${max}) grey levels`);

    const onOpencvCrop = fc.cosine(await embed(Uint8ClampedArray.from(r.aligned_rgb)), r.embedding);
    check(onOpencvCrop > 0.9999, `${name}: SFace in ONNX Runtime Web on OpenCV's crop, cosine ${onOpencvCrop.toFixed(6)}`);
    const full = fc.cosine(await embed(fc.warpAffine(img, fc.similarityMatrix(primary))), r.embedding);
    check(full > 0.995, `${name}: full mobile pipeline vs OpenCV embedding, cosine ${full.toFixed(5)}`);
    // the yaw proxy uses the eye and nose landmarks of this row
    const yaw = (row) => ((row[8] - (row[4] + row[6]) / 2) / Math.hypot(row[6] - row[4], row[7] - row[5]));
    check(Math.abs(yaw(primary) - r.yaw) < 0.01, `${name}: yaw proxy ${yaw(primary).toFixed(4)} vs ${r.yaw.toFixed(4)}`);
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
console.log(failures.length ? `${failures.length} FAILED` : "ALL PASSED");
process.exitCode = failures.length ? 1 : 0;
