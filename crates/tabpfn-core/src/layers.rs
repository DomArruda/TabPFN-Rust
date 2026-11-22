//! Low-level Candle layers: RMSNorm, RoPE, MLPs, Fourier embedders.
//! `VarBuilder::get` signature in candle 0.10 is `get(shape, name)`.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

/// RMSNorm over the last dim. eps = f32::EPSILON (torch default).
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn load_dim(vb: &VarBuilder, name: &str, dim: usize) -> crate::Result<Self> {
        let w = vb.get(dim, name)?;
        Ok(Self { weight: w, eps: f32::EPSILON as f64 })
    }
    pub fn from_weight(weight: Tensor) -> Self {
        Self { weight, eps: f32::EPSILON as f64 }
    }
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let sq = x.sqr()?;
        let ms = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (ms + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&rms)?;
        Ok(normed.broadcast_mul(&self.weight)?)
    }
}

pub struct LayerNormFlat {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormFlat {
    pub fn load(vb: &VarBuilder, prefix: &str, dim: usize) -> crate::Result<Self> {
        Ok(Self {
            weight: vb.get(dim, &format!("{prefix}.weight"))?,
            bias: vb.get(dim, &format!("{prefix}.bias"))?,
            eps: 1e-5,
        })
    }
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        crate::log::layer_norm_forward(x, &self.weight, &self.bias, self.eps)
    }
}

pub fn gelu(x: &Tensor) -> crate::Result<Tensor> {
    Ok(x.gelu_erf()?)
}

pub fn linear_no_bias(x: &Tensor, w: &Tensor) -> crate::Result<Tensor> {
    // y = x @ W^T where W is (out, in). Flatten leading dims for batched matmul.
    let wt = w.transpose(0, 1)?; // (in, out)
    let dims = x.dims().to_vec();
    if dims.len() <= 2 {
        Ok(x.matmul(&wt)?)
    } else {
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.reshape((lead, dims[dims.len() - 1]))?;
        let out = flat.matmul(&wt)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(w.dim(0)?);
        Ok(out.reshape(out_dims.as_slice())?)
    }
}

pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
    let wt = w.transpose(0, 1)?;
    let dims = x.dims().to_vec();
    let out = if dims.len() <= 2 {
        x.matmul(&wt)?
    } else {
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.reshape((lead, dims[dims.len() - 1]))?;
        let o = flat.matmul(&wt)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(w.dim(0)?);
        o.reshape(out_dims.as_slice())?
    };
    Ok(out.broadcast_add(b)?)
}

pub struct Mlp {
    pub w1: Tensor,
    pub w2: Tensor,
}

impl Mlp {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, hidden: usize) -> crate::Result<Self> {
        let w1 = vb.get((hidden, emsize), &format!("{prefix}.0.weight"))?;
        let w2 = vb.get((emsize, hidden), &format!("{prefix}.2.weight"))?;
        Ok(Self { w1, w2 })
    }
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let h = linear_no_bias(x, &self.w1)?;
        let h = gelu(&h)?;
        Ok(linear_no_bias(&h, &self.w2)?)
    }
}

/// RotaryEmbedding: dim=head_dim, theta=100_000, split-half rotation, f32.
pub struct RotaryEmbedding {
    pub inv_freq: Tensor,
    pub dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f32, device: &Device) -> crate::Result<Self> {
        let mut inv = Vec::with_capacity(dim / 2);
        for i in 0..dim / 2 {
            inv.push(1.0f32 / theta.powf(2.0 * i as f32 / dim as f32));
        }
        Ok(Self { inv_freq: Tensor::from_vec(inv, (dim / 2,), device)?, dim })
    }

    /// t: (..., S, D). Returns rotated tensor.
    pub fn apply(&self, t: &Tensor) -> crate::Result<Tensor> {
        let dims = t.dims();
        let (s, d) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        let positions: Vec<f32> = (0..s).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(positions, (s, 1), t.device())?;
        let freqs = pos.broadcast_mul(&self.inv_freq.reshape((1, self.dim / 2))?)?;
        let cos2d = freqs.cos()?;
        let sin2d = freqs.sin()?;
        // cat along last dim: (S, D)
        let cos = Tensor::cat(&[&cos2d, &cos2d], 1)?;
        let sin = Tensor::cat(&[&sin2d, &sin2d], 1)?;
        let mut shape = vec![1usize; dims.len()];
        shape[dims.len() - 2] = s;
        shape[dims.len() - 1] = d;
        let cos = cos.reshape(shape.clone())?;
        let sin = sin.reshape(shape)?;
        let half = d / 2;
        let first = t.narrow(dims.len() - 1, 0, half)?;
        let second = t.narrow(dims.len() - 1, half, half)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], dims.len() - 1)?;
        let lhs = t.broadcast_mul(&cos)?;
        let rhs = rotated.broadcast_mul(&sin)?;
        Ok(lhs.broadcast_add(&rhs)?)
    }
}

pub struct SoftmaxScalingMlp {
    pub base_w1: Tensor,
    pub base_b1: Tensor,
    pub base_w2: Tensor,
    pub base_b2: Tensor,
    pub query_w1: Tensor,
    pub query_b1: Tensor,
    pub query_w2: Tensor,
    pub query_b2: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl SoftmaxScalingMlp {
    pub fn load(vb: &VarBuilder, prefix: &str, num_heads: usize, head_dim: usize, hidden: usize) -> crate::Result<Self> {
        let base_out = num_heads * head_dim;
        let get = |n: &str, s: (usize, usize)| vb.get(s, &format!("{prefix}.{n}"));
        let get1 = |n: &str, s: usize| vb.get(s, &format!("{prefix}.{n}"));
        Ok(Self {
            base_w1: get("base_mlp.0.weight", (hidden, 1))?,
            base_b1: get1("base_mlp.0.bias", hidden)?,
            base_w2: get("base_mlp.2.weight", (base_out, hidden))?,
            base_b2: get1("base_mlp.2.bias", base_out)?,
            query_w1: get("query_mlp.0.weight", (hidden, head_dim))?,
            query_b1: get1("query_mlp.0.bias", hidden)?,
            query_w2: get("query_mlp.2.weight", (head_dim, hidden))?,
            query_b2: get1("query_mlp.2.bias", head_dim)?,
            num_heads,
            head_dim,
        })
    }

    /// q: (B, S, H, D), n = KV length.
    pub fn forward(&self, q: &Tensor, n: usize) -> crate::Result<Tensor> {
        let logn = (n.max(1) as f32).ln();
        let logn_t = Tensor::from_vec(vec![logn], (1, 1), q.device())?;
        let h1 = linear(&logn_t, &self.base_w1, &self.base_b1)?;
        let h1 = gelu(&h1)?;
        let base = linear(&h1, &self.base_w2, &self.base_b2)?;
        let base = base.reshape((1, 1, self.num_heads, self.head_dim))?;
        let mod_in = linear(q, &self.query_w1, &self.query_b1)?;
        let mod_in = gelu(&mod_in)?;
        let mod_out = linear(&mod_in, &self.query_w2, &self.query_b2)?;
        let scales = base.broadcast_mul(&(mod_out.tanh()? + 1.0)?)?;
        Ok(q.broadcast_mul(&scales)?)
    }
}

pub struct FourierEmbedder {
    pub frequencies: Tensor,
    pub in_w: Tensor,
}

impl FourierEmbedder {
    pub fn load(vb: &VarBuilder, prefix: &str, g: usize, f: usize, e: usize) -> crate::Result<Self> {
        Ok(Self {
            frequencies: vb.get((g, f), &format!("{prefix}.frequencies"))?,
            in_w: vb.get((e, 2 * f), &format!("{prefix}.in_linear.weight"))?,
        })
    }

    /// x: (..., G) -> (..., E)
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let dims = x.dims().to_vec();
        let g = dims[dims.len() - 1];
        let f = self.frequencies.dim(1)?;
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let proj = x.reshape((lead, g, 1))?.broadcast_mul(&self.frequencies.reshape((1, g, f))?)?;
        let feats = Tensor::cat(&[&proj.sin()?, &proj.cos()?], 2)?; // (lead, G, 2F)
        let summed = feats.sum(1)?; // sum over G -> (lead, 2F)
        let out = linear_no_bias(&summed, &self.in_w)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.in_w.dim(0)?);
        Ok(out.reshape(out_dims.as_slice())?)
    }
}
