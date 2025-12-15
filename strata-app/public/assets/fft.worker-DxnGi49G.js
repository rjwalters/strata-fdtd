function y(s, t) {
  const o = new Array(t).fill(0);
  for (let e = 0; e < Math.min(s.length, t); e++)
    o[e] = s[e];
  const r = Math.log2(t), c = new Float32Array(t), a = new Float32Array(t);
  for (let e = 0; e < t; e++) {
    const u = q(e, r);
    c[u] = o[e];
  }
  for (let e = 2; e <= t; e *= 2) {
    const u = e / 2, n = -2 * Math.PI / e;
    for (let i = 0; i < t; i += e)
      for (let l = 0; l < u; l++) {
        const d = n * l, g = Math.cos(d), p = Math.sin(d), h = i + l, m = i + l + u, w = g * c[m] - p * a[m], M = p * c[m] + g * a[m];
        c[m] = c[h] - w, a[m] = a[h] - M, c[h] = c[h] + w, a[h] = a[h] + M;
      }
  }
  const f = new Float32Array(t * 2);
  for (let e = 0; e < t; e++)
    f[2 * e] = c[e], f[2 * e + 1] = a[e];
  return f;
}
function q(s, t) {
  let o = 0;
  for (let r = 0; r < t; r++)
    o = o << 1 | s & 1, s >>= 1;
  return o;
}
function A(s) {
  return Math.pow(2, Math.ceil(Math.log2(s)));
}
function F(s, t, o) {
  const r = o ?? A(s.length), c = new Array(r).fill(0);
  for (let n = 0; n < Math.min(s.length, r); n++) {
    const i = 0.5 * (1 - Math.cos(2 * Math.PI * n / (s.length - 1)));
    c[n] = s[n] * i;
  }
  const a = y(c, r), f = r / 2, e = new Float32Array(f), u = new Float32Array(f);
  for (let n = 0; n < f; n++) {
    const i = a[2 * n], l = a[2 * n + 1];
    e[n] = n * t / r, u[n] = Math.sqrt(i * i + l * l) / r;
  }
  return { frequencies: e, magnitude: u };
}
self.onmessage = (s) => {
  const t = s.data;
  try {
    switch (t.type) {
      case "computeSpectrum": {
        const o = F(
          t.data,
          t.sampleRate,
          t.nfft
        ), r = {
          type: "computeSpectrum",
          id: t.id,
          frequencies: o.frequencies,
          magnitude: o.magnitude
        };
        self.postMessage(r, {
          transfer: [o.frequencies.buffer, o.magnitude.buffer]
        });
        break;
      }
    }
  } catch (o) {
    const r = {
      type: "error",
      id: t.id,
      error: o instanceof Error ? o.message : "Unknown error"
    };
    self.postMessage(r);
  }
};
//# sourceMappingURL=fft.worker-DxnGi49G.js.map
