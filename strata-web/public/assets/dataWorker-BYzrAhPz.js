const D = {
  targetVoxels: 262144,
  // 64³ - good balance of quality and performance
  method: "average"
};
function E(n, e) {
  const s = n[0] * n[1] * n[2];
  return s <= e ? 1 : Math.ceil(Math.pow(s / e, 1 / 3));
}
function O(n, e, s = {}) {
  const o = { ...D, ...s }, [r, l, a] = e;
  if (r * l * a <= o.targetVoxels)
    return {
      data: n,
      shape: e,
      factor: 1,
      originalShape: e
    };
  const t = E(e, o.targetVoxels), V = Math.ceil(r / t), b = Math.ceil(l / t), M = Math.ceil(a / t), X = [V, b, M], Y = V * b * M, v = new Float32Array(Y), k = (i, f, d) => i * l * a + f * a + d, Z = (i, f, d) => i * b * M + f * M + d;
  for (let i = 0; i < V; i++)
    for (let f = 0; f < b; f++)
      for (let d = 0; d < M; d++) {
        const x = i * t, g = f * t, y = d * t, F = Math.min(x + t, r), N = Math.min(g + t, l), P = Math.min(y + t, a);
        let A;
        switch (o.method) {
          case "nearest": {
            const p = Math.min(x + Math.floor(t / 2), r - 1), h = Math.min(g + Math.floor(t / 2), l - 1), c = Math.min(y + Math.floor(t / 2), a - 1);
            A = n[k(p, h, c)];
            break;
          }
          case "max": {
            let p = 0, h = 0;
            for (let c = x; c < F; c++)
              for (let m = g; m < N; m++)
                for (let w = y; w < P; w++) {
                  const S = n[k(c, m, w)], z = Math.abs(S);
                  z > p && (p = z, h = S);
                }
            A = h;
            break;
          }
          case "average":
          default: {
            let p = 0, h = 0;
            for (let c = x; c < F; c++)
              for (let m = g; m < N; m++)
                for (let w = y; w < P; w++)
                  p += n[k(c, m, w)], h++;
            A = h > 0 ? p / h : 0;
            break;
          }
        }
        v[Z(i, f, d)] = A;
      }
  return {
    data: v,
    shape: X,
    factor: t,
    originalShape: e
  };
}
function T(n, e, s) {
  const o = e * s;
  let r = 0;
  for (let t = 0; t < n.length; t++)
    Math.abs(n[t]) >= o && r++;
  const l = new Uint32Array(r), a = new Float32Array(r);
  let u = 0;
  for (let t = 0; t < n.length; t++)
    Math.abs(n[t]) >= o && (l[u] = t, a[u] = n[t], u++);
  return { indices: l, values: a, count: r };
}
self.onmessage = (n) => {
  const e = n.data;
  try {
    switch (e.type) {
      case "downsample": {
        const s = O(
          e.data,
          e.shape,
          e.options
        ), o = {
          type: "downsample",
          id: e.id,
          result: s
        };
        self.postMessage(o, { transfer: [s.data.buffer] });
        break;
      }
      case "filter": {
        const s = T(
          e.data,
          e.threshold,
          e.maxPressure
        ), o = {
          type: "filter",
          id: e.id,
          result: s
        };
        self.postMessage(o, {
          transfer: [s.indices.buffer, s.values.buffer]
        });
        break;
      }
      case "downsampleAndFilter": {
        const s = O(
          e.data,
          e.shape,
          e.downsampleOptions
        );
        let o = 0;
        for (let a = 0; a < s.data.length; a++) {
          const u = Math.abs(s.data[a]);
          u > o && (o = u);
        }
        const r = T(
          s.data,
          e.threshold,
          o
        ), l = {
          type: "downsampleAndFilter",
          id: e.id,
          downsampleResult: s,
          filterResult: r
        };
        self.postMessage(l, {
          transfer: [
            s.data.buffer,
            r.indices.buffer,
            r.values.buffer
          ]
        });
        break;
      }
    }
  } catch (s) {
    const o = {
      type: "error",
      id: e.id,
      error: s instanceof Error ? s.message : "Unknown error"
    };
    self.postMessage(o);
  }
};
//# sourceMappingURL=dataWorker-BYzrAhPz.js.map
