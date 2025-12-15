function S(o, e, s) {
  const t = o.length, r = 1e3 / s;
  if (e >= t || e < 3) {
    const n = new Float32Array(t * 2);
    for (let c = 0; c < t; c++)
      n[c * 2] = c * r, n[c * 2 + 1] = o[c];
    return n;
  }
  const f = [], u = (t - 2) / (e - 2);
  f.push(0 * r, o[0]);
  let l = 0;
  for (let n = 0; n < e - 2; n++) {
    const c = Math.floor((n + 1) * u) + 1, i = Math.floor((n + 2) * u) + 1, g = Math.min(i, t - 1), h = Math.floor((n + 2) * u) + 1, M = Math.floor((n + 3) * u) + 1, a = Math.min(M, t);
    let b = 0, p = 0, x = 0;
    for (let m = h; m < a; m++)
      b += m * r, p += o[m], x++;
    x > 0 ? (b /= x, p /= x) : (b = (t - 1) * r, p = o[t - 1]);
    const w = l * r, d = o[l];
    let y = -1, k = c;
    for (let m = c; m < g; m++) {
      const q = m * r, A = o[m], F = Math.abs(
        (w - b) * (A - d) - (w - q) * (p - d)
      );
      F > y && (y = F, k = m);
    }
    f.push(k * r, o[k]), l = k;
  }
  return f.push((t - 1) * r, o[t - 1]), new Float32Array(f);
}
function B(o, e, s) {
  const t = o.length, r = 1e3 / s;
  if (e >= t || e < 2) {
    const l = new Float32Array(t * 2);
    for (let n = 0; n < t; n++)
      l[n * 2] = n * r, l[n * 2 + 1] = o[n];
    return l;
  }
  const f = t / e, u = [];
  for (let l = 0; l < e; l++) {
    const n = Math.floor(l * f), c = Math.min(Math.floor((l + 1) * f), t);
    let i = o[n], g = o[n], h = n, M = n;
    for (let a = n; a < c; a++)
      o[a] < i && (i = o[a], h = a), o[a] > g && (g = o[a], M = a);
    h <= M ? (u.push(h * r, i), h !== M && u.push(M * r, g)) : (u.push(M * r, g), u.push(h * r, i));
  }
  return new Float32Array(u);
}
function E(o, e, s, t, r) {
  const f = Math.log10(Math.max(t, 1)), l = (Math.log10(r) - f) / s, n = new Float32Array(s), c = new Float32Array(s);
  for (let i = 0; i < s; i++) {
    const g = Math.pow(10, f + i * l), h = Math.pow(10, f + (i + 1) * l), M = Math.sqrt(g * h);
    let a = 0, b = !1;
    for (let p = 0; p < o.length; p++)
      o[p] >= g && o[p] < h && (b = !0, e[p] > a && (a = e[p]));
    n[i] = M, c[i] = b ? a : 0;
  }
  return { frequencies: n, magnitude: c };
}
self.onmessage = (o) => {
  const e = o.data;
  try {
    switch (e.type) {
      case "lttb": {
        const s = S(
          e.data,
          e.targetPoints,
          e.sampleRate
        ), t = {
          type: "lttb",
          id: e.id,
          points: s,
          length: s.length / 2
        };
        self.postMessage(t, { transfer: [s.buffer] });
        break;
      }
      case "minmax": {
        const s = B(
          e.data,
          e.targetPoints,
          e.sampleRate
        ), t = {
          type: "minmax",
          id: e.id,
          points: s,
          length: s.length / 2
        };
        self.postMessage(t, { transfer: [s.buffer] });
        break;
      }
      case "logbin": {
        const s = E(
          e.frequencies,
          e.magnitude,
          e.targetBins,
          e.minFreq,
          e.maxFreq
        ), t = {
          type: "logbin",
          id: e.id,
          frequencies: s.frequencies,
          magnitude: s.magnitude
        };
        self.postMessage(t, {
          transfer: [s.frequencies.buffer, s.magnitude.buffer]
        });
        break;
      }
    }
  } catch (s) {
    const t = {
      type: "error",
      id: e.id,
      error: s instanceof Error ? s.message : "Unknown error"
    };
    self.postMessage(t);
  }
};
//# sourceMappingURL=downsample.worker-D_BwNp-0.js.map
