// WaveLock mobile: the face worker (module worker).
//
// Runs the desktop's face models with ONNX Runtime Web: YuNet (detection,
// 0.2 MB) and SFace (128-value identity embedding, 38.7 MB, loaded only when
// a face is enrolled or captured). Pre- and post-processing are
// face_core.js, checked against OpenCV by tests/face_parity.mjs.
//
// Messages: {id, type: "detect", image} -> {row}            (primary face or null)
//           {id, type: "embed", image, row} -> {embedding}
//           {id, type: "detect_embed", image} -> {row, embedding}
// image = {data: RGBA bytes, width, height}, at most 640 px per side.

import * as ort from "../../vendor/ort/ort.wasm.min.mjs";
import * as fc from "./face_core.js";

const BASE = new URL("../../", import.meta.url).href;
ort.env.wasm.wasmPaths = BASE + "vendor/ort/";
ort.env.wasm.numThreads = 1;          // no cross-origin isolation needed
ort.env.logLevel = "error";

let yunet = null;
let sface = null;

async function session(name) {
  const res = await fetch(BASE + "models/" + name);
  if (!res.ok) throw new Error(`model ${name}: HTTP ${res.status}`);
  return ort.InferenceSession.create(new Uint8Array(await res.arrayBuffer()), { executionProviders: ["wasm"] });
}

async function detect(image) {
  yunet = yunet || await session("face_detection_yunet_2023mar.onnx");
  const out = await yunet.run({ input: new ort.Tensor("float32", fc.yunetInput(image), [1, 3, 640, 640]) });
  const plain = Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.data]));
  return fc.primaryFace(fc.yunetDecode(plain));
}

async function embed(image, row) {
  sface = sface || await session("face_recognition_sface_2021dec.onnx");
  const rgb = fc.warpAffine(image, fc.similarityMatrix(row));
  const out = await sface.run({ data: new ort.Tensor("float32", fc.sfaceInput(rgb), [1, 3, 112, 112]) });
  return Array.from(out.fc1.data);
}

let queue = Promise.resolve();
self.onmessage = (e) => {
  queue = queue.then(async () => {
    const { id, type, image, row } = e.data;
    try {
      const img = { data: new Uint8ClampedArray(image.data), width: image.width, height: image.height };
      if (type === "detect") {
        self.postMessage({ id, result: { row: await detect(img) } });
      } else if (type === "embed") {
        self.postMessage({ id, result: { embedding: await embed(img, row) } });
      } else if (type === "detect_embed") {
        const r = await detect(img);
        self.postMessage({ id, result: { row: r, embedding: r ? await embed(img, r) : null } });
      } else if (type === "warm") {
        if (e.data.sface) sface = sface || await session("face_recognition_sface_2021dec.onnx");
        yunet = yunet || await session("face_detection_yunet_2023mar.onnx");
        self.postMessage({ id, result: { ok: true } });
      }
    } catch (err) {
      self.postMessage({ id, error: String((err && err.message) || err) });
    }
  });
};
