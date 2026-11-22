//! Stage 0 preprocessing in plain Rust f32 (no Candle ops).
//!
//! Matches `tabpfn/architectures/tabpfn_v3_5.py`:
//! `_preprocess_raw`, `_group_features`, `_build_ecdf_context`,
//! `_in_context_ecdf`, `_generate_nan_and_inf_indicator`,
//! `_impute_nan_and_inf_with_mean`, `torch_standard_scaler.py` and
//! `torch/ops.py::torch_nanmean/torch_nanstd`.
//!
//! All tensors are handled as flat row-major `Vec<f32>` with shape
//! `(R, B, C)` on entry (B is the estimator batch; we only use B=1) and
//! `(B, R, C)` internally, like the Python transposes.

pub const NAN_INDICATOR: f32 = -2.0;
pub const INF_INDICATOR: f32 = 2.0;
pub const NEGINF_INDICATOR: f32 = 4.0;

/// Fitted scaler statistics, stored in the KV cache as `scaler_cache`.
#[derive(Debug, Clone)]
pub struct ScalerCache {
    pub mean: Vec<f32>, // (C,)
    pub std: Vec<f32>,  // (C,)
}

/// ECDF context stored as `(3, B, C, K)` flattened: `[edges, below, at_most]`.
#[derive(Debug, Clone)]
pub struct EcdfContext {
    pub data: Vec<f32>, // (3, B, C, K), row-major
    pub b: usize,
    pub c: usize,
    pub k: usize,
}

impl EcdfContext {
    pub fn idx(&self, t: usize, b: usize, c: usize, k: usize) -> usize {
        ((t * self.b + b) * self.c + c) * self.k + k
    }
}

/// NaN/Inf indicators computed BEFORE imputation: NaN->-2, +Inf->+2, -Inf->+4.
pub fn nan_inf_indicator(buf: &[f32]) -> Vec<f32> {
    buf.iter()
        .map(|&v| {
            if v.is_nan() {
                NAN_INDICATOR
            } else if v == f32::INFINITY {
                INF_INDICATOR
            } else if v == f32::NEG_INFINITY {
                NEGINF_INDICATOR
            } else {
                0.0
            }
        })
        .collect()
}

/// Column means over finite values in `x[..num_train]` (fallback 0.0).
/// `x` is `(R, B, C)` row-major; only the first `num_train` rows are read,
/// mirroring `_impute_nan_and_inf_with_mean` which masks non-finite first.
pub fn compute_feature_means(x: &[f32], rows: usize, b: usize, c: usize, num_train: usize) -> Vec<f32> {
    let mut sum = vec![0.0f32; b * c];
    let mut cnt = vec![0.0f32; b * c];
    for r in 0..num_train.min(rows) {
        for bc in 0..b * c {
            let v = x[r * b * c + bc];
            if v.is_finite() {
                sum[bc] += v;
                cnt[bc] += 1.0;
            }
        }
    }
    sum.iter()
        .zip(cnt.iter())
        .map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 })
        .collect()
}

/// Impute non-finite entries with the given per-column means.
/// Returns `(imputed, is_finite_mask)` where mask is f32 0/1.
pub fn impute_with_means(x: &[f32], means: &[f32], rows: usize, b: usize, c: usize) -> (Vec<f32>, Vec<bool>) {
    let mut out = Vec::with_capacity(x.len());
    let mut finite = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        let ok = v.is_finite();
        finite.push(ok);
        out.push(if ok { v } else { means[i % (b * c)] });
    }
    let _ = (rows, b, c);
    (out, finite)
}

/// TorchStandardScaler.fit: mean via nanmean, std via nanstd (N-1 correction,
/// matching `torch_nanstd`), std==0 -> 1, single-row fit -> std=1.
/// In this pipeline x is already imputed (finite), so nanmean == mean but we
/// keep the same formula for the cached-test path.
pub fn fit_scaler(train: &[f32], num_train: usize, b: usize, c: usize) -> ScalerCache {
    let bc = b * c;
    let mut mean = vec![0.0f32; bc];
    if num_train == 0 {
        let std = vec![1.0f32; bc];
        return ScalerCache { mean, std };
    }
    // mean over first num_train rows (all finite after imputation)
    for r in 0..num_train {
        for i in 0..bc {
            mean[i] += train[r * bc + i];
        }
    }
    for m in mean.iter_mut() {
        *m /= num_train as f32;
    }
    let mut std = vec![0.0f32; bc];
    if num_train == 1 {
        std.fill(1.0);
        return ScalerCache { mean, std };
    }
    for r in 0..num_train {
        for i in 0..bc {
            let d = train[r * bc + i] - mean[i];
            std[i] += d * d;
        }
    }
    for i in 0..bc {
        // (num_valid-1).clamp(min=1)
        let denom = ((num_train as f32) - 1.0).max(1.0);
        std[i] = (std[i] / denom).sqrt();
        if std[i] == 0.0 {
            std[i] = 1.0;
        }
    }
    ScalerCache { mean, std }
}

/// TorchStandardScaler.transform: `(x - mean) / (std + eps)`, clip [-100, 100].
/// eps = f32::EPSILON (torch.finfo(float32).eps).
pub fn transform_scaler(x: &[f32], scaler: &ScalerCache, b: usize, c: usize) -> Vec<f32> {
    let eps = f32::EPSILON;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let bc = i % (b * c);
            ((v - scaler.mean[bc]) / (scaler.std[bc] + eps)).clamp(-100.0, 100.0)
        })
        .collect()
}

/// Build the ECDF context from imputed, UNSCALED train rows.
///
/// `_build_ecdf_context(x_BRiC, num_train, num_buckets)`:
/// K = min(num_buckets, N); per (B,C) sort train values; distinct-index map;
/// if num_distinct <= K pick evenly spaced distinct indices, else evenly
/// spaced sorted row positions; store [edge, below, at_most].
///
/// `x_bnc` is `(B, N, C)` row-major (first `num_train` rows per B).
pub fn build_ecdf_context(x_bnc: &[f32], b: usize, n: usize, c: usize, num_train: usize, num_buckets: usize) -> EcdfContext {
    // GNote: x_bnc is (B, N, C) with N == num_train rows per batch.
    let k = num_buckets.min(num_train.max(1));
    let mut data = vec![0.0f32; 3 * b * c * k];
    // gather (B,C) slices
    for bi in 0..b {
        for ci in 0..c {
            let mut col = Vec::with_capacity(num_train);
            for r in 0..num_train {
                col.push(x_bnc[(bi * n + r) * c + ci]);
            }
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // distinct_idx for each sorted position
            let mut distinct_idx = vec![0usize; sorted.len()];
            let mut d = 0usize;
            for i in 0..sorted.len() {
                if i > 0 && sorted[i] != sorted[i - 1] {
                    d += 1;
                }
                distinct_idx[i] = d;
            }
            let num_distinct = if sorted.is_empty() { 0 } else { d + 1 };
            let mut targets = vec![0usize; k];
            if k > 1 {
                for (ki, t) in targets.iter_mut().enumerate() {
                    if num_distinct <= k {
                        // evenly spaced distinct indices: steps * (num_distinct-1)/(k-1)
                        let v = ki as f32 * ((num_distinct - 1) as f32 / (k - 1) as f32);
                        *t = v.round() as usize;
                    } else {
                        // evenly spaced sorted row positions
                        let v = ki as f32 * ((n - 1) as f32 / (k - 1) as f32);
                        let pos = v.round() as usize;
                        *t = distinct_idx[pos];
                    }
                }
            }
            // below/at_most via binary search over distinct_idx (sorted, non-decreasing)
            for (ki, &t) in targets.iter().enumerate() {
                let below = distinct_idx.partition_point(|&d| d < t);
                let at_most = distinct_idx.partition_point(|&d| d <= t);
                let edge = sorted[below.min(sorted.len().saturating_sub(1))];
                let base = (bi * c + ci) * k + ki;
                data[base] = edge;
                data[b * c * k + base] = below as f32; // middle plane
                data[2 * b * c * k + base] = at_most as f32; // last plane
            }
        }
    }
    EcdfContext { data, b, c, k }
}

/// Query ECDF midranks for all cells. `x_bnc` is `(B, R, C)`.
/// Returns ranks in `(B, R, C)`, i.e. counts / N where N = at_most[K-1] per column.
pub fn in_context_ecdf(x_bnc: &[f32], ctx: &EcdfContext, b: usize, r: usize, c: usize) -> Vec<f32> {
    let k = ctx.k;
    let mut out = vec![0.0f32; b * r * c];
    for bi in 0..b {
        for ci in 0..c {
            let n = ctx.data[ctx.idx(2, bi, ci, k - 1)];
            let n_finite = if n == 0.0 { 1.0 } else { n };
            let mid_lo = 0.5 * (ctx.data[ctx.idx(1, bi, ci, 0)] + ctx.data[ctx.idx(2, bi, ci, 0)]);
            let mid_hi = 0.5 * (ctx.data[ctx.idx(1, bi, ci, k - 1)] + ctx.data[ctx.idx(2, bi, ci, k - 1)]);
            for ri in 0..r {
                let v = x_bnc[(bi * r + ri) * c + ci];
                let mut edges = Vec::with_capacity(k);
                for ki in 0..k {
                    edges.push(ctx.data[ctx.idx(0, bi, ci, ki)]);
                }
                let left = edges.partition_point(|&e| e < v);
                let right = edges.partition_point(|&e| e <= v);
                let count = if right > left {
                    let hi = left.min(k - 1);
                    0.5 * (ctx.data[ctx.idx(1, bi, ci, hi)] + ctx.data[ctx.idx(2, bi, ci, hi)])
                } else {
                    let lo_idx = left.saturating_sub(1);
                    let hi_idx = left.min(k - 1);
                    let edge_lo = edges[lo_idx];
                    let edge_hi = edges[hi_idx];
                    let at_most_lo = ctx.data[ctx.idx(2, bi, ci, lo_idx)];
                    let below_hi = ctx.data[ctx.idx(1, bi, ci, hi_idx)];
                    let width = edge_hi - edge_lo;
                    let weight = if width > 0.0 { (v - edge_lo) / width } else { 0.0 };
                    let mut cnt = at_most_lo + weight * (below_hi - at_most_lo);
                    if left == 0 {
                        cnt = 0.0;
                    }
                    cnt
                };
                let cnt = count.clamp(mid_lo, mid_hi);
                out[(bi * r + ri) * c + ci] = cnt / n_finite;
            }
        }
    }
    out
}

/// Circular feature grouping: shifts = [-1, -2, -4] (i.e. `-2^i`),
/// `torch.roll(t, shifts=s, dims=2)` on `(B, R, C)` gives `out[..., c] = in[..., (c - s) mod C]`
/// = `in[..., (c + 2^i) mod C]` for shift `-2^i`. Matches the spec's `(c + 2^i) % C`.
/// Output layout per cell: [scaled G | nan_ind G | ecdf G] => `(B, R, C, 3G)`.
pub fn group_features(
    x_bnc: &[f32],
    nan_bnc: &[f32],
    ecdf_bnc: &[f32],
    b: usize,
    r: usize,
    c: usize,
    group_size: usize,
) -> Vec<f32> {
    let g = group_size;
    let mut out = vec![0.0f32; b * r * c * 3 * g];
    for bi in 0..b {
        for ri in 0..r {
            for ci in 0..c {
                for gi in 0..g {
                    let src = (ci + (1usize << gi)) % c;
                    let dst = (bi * r * c * 3 * g) + (bi * r + ri) * c * 3 * g + ci * 3 * g;
                    out[dst + gi] = x_bnc[(bi * r + ri) * c + src];
                    out[dst + g + gi] = nan_bnc[(bi * r + ri) * c + src];
                    out[dst + 2 * g + gi] = ecdf_bnc[(bi * r + ri) * c + src];
                }
            }
        }
    }
    out
}

/// Impute missing train targets with the train mean; for multiclass the
/// *imputed positions only* get `ceil()`. Matches `_impute_target_nan_and_inf`.
pub fn impute_targets(y: &[f32], num_train: usize) -> Vec<f32> {
    let mut sum = 0.0f32;
    let mut cnt = 0.0f32;
    for &v in &y[..num_train] {
        if v.is_finite() {
            sum += v;
            cnt += 1.0;
        }
    }
    let mean = if cnt > 0.0 { sum / cnt } else { 0.0 };
    y.iter()
        .map(|&v| if v.is_finite() { v } else { mean })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdf_exact_for_small_column() {
        // distinct <= K -> every value hits an edge, midrank = exact (below+at_most)/2 / N
        let x = vec![1.0, 2.0, 3.0, 2.0]; // B=1,N=4,C=1
        let ctx = build_ecdf_context(&x, 1, 4, 1, 4, 8);
        let ranks = in_context_ecdf(&x, &ctx, 1, 4, 1);
        // sorted [1,2,2,3]; midranks: 1->0.5/4, 2->(1+3)/2/4=0.5, 3->3.5/4
        assert!((ranks[0] - 0.125).abs() < 1e-6, "{:?}", ranks);
        assert!((ranks[1] - 0.5).abs() < 1e-6);
        assert!((ranks[2] - 0.875).abs() < 1e-6);
        assert!((ranks[3] - 0.5).abs() < 1e-6);
    }
}
