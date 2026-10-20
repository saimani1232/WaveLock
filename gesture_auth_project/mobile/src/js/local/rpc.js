// Promise wrapper around a worker that answers {id, result} or {id, error}
// (and may send {progress} messages while it starts).

export class Rpc {
  constructor(worker, onProgress = null) {
    this.worker = worker;
    this.next = 1;
    this.pending = new Map();
    this.onProgress = onProgress;
    worker.onmessage = (e) => {
      const m = e.data;
      if (m && m.progress !== undefined && m.id === undefined) {
        if (this.onProgress) this.onProgress(m.progress);
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.error !== undefined) p.reject(new Error(m.error));
      else p.resolve(m.result);
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || "worker failed");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  call(message) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id });
    });
  }
}
