//! Stage modules mirroring `tabpfn/architectures/tabpfn_v3_5.py` classes.

use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::attention::sdpa;
use crate::layers::{linear_no_bias, Mlp, RmsNorm, RotaryEmbedding, SoftmaxScalingMlp};

pub struct CrossAttention {
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub out_w: Tensor,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scaling: Option<SoftmaxScalingMlp>,
}

impl CrossAttention {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, scaling: Option<SoftmaxScalingMlp>) -> crate::Result<Self> {
        let head_dim = emsize / num_heads;
        let hd = head_dim * num_heads;
        let g = |n: &str, s: (usize, usize)| vb.get(s, &format!("{prefix}.attn.{n}.weight"));
        let _g1 = |n: &str, s: usize| vb.get(s, &format!("{prefix}.attn.{n}.weight"));
        Ok(Self {
            q_w: g("q_projection", (hd, emsize))?,
            k_w: g("k_projection", (hd, emsize))?,
            v_w: g("v_projection", (hd, emsize))?,
            out_w: vb.get((emsize, hd), &format!("{prefix}.attn.out_projection.weight"))?,
            q_norm: RmsNorm::load_dim(vb, &format!("{prefix}.attn.q_norm.weight"), head_dim)?,
            k_norm: RmsNorm::load_dim(vb, &format!("{prefix}.attn.k_norm.weight"), head_dim)?,
            num_heads,
            head_dim,
            scaling,
        })
    }

    pub fn forward(&self, x_q: &Tensor, x_kv: &Tensor) -> crate::Result<Tensor> {
        let (b, q_len) = (x_q.dim(0)?, x_q.dim(1)?);
        let v_len = x_kv.dim(1)?;
        let h = self.num_heads;
        let d = self.head_dim;
        let q = linear_no_bias(x_q, &self.q_w)?.reshape((b, q_len, h, d))?;
        let k = linear_no_bias(x_kv, &self.k_w)?.reshape((b, v_len, h, d))?;
        let v = linear_no_bias(x_kv, &self.v_w)?.reshape((b, v_len, h, d))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let out = crate::attention::sdpa_scaled(&q, &k, &v, self.scaling.as_ref())?;
        let out = out.reshape((b, q_len, h * d))?;
        Ok(linear_no_bias(&out, &self.out_w)?)
    }
}

pub struct CrossAttentionBlock {
    pub attn: CrossAttention,
    pub mlp: Mlp,
    pub ln_q_w: Tensor,
    pub ln_kv_w: Tensor,
    pub ln2_w: Tensor,
}

impl CrossAttentionBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, scaling: Option<SoftmaxScalingMlp>) -> crate::Result<Self> {
        let hidden = emsize * 2;
        Ok(Self {
            attn: CrossAttention::load(vb, prefix, emsize, num_heads, scaling)?,
            mlp: Mlp::load(vb, &format!("{prefix}.mlp"), emsize, hidden)?,
            ln_q_w: vb.get(emsize, &format!("{prefix}.layernorm_q.weight"))?,
            ln_kv_w: vb.get(emsize, &format!("{prefix}.layernorm_kv.weight"))?,
            ln2_w: vb.get(emsize, &format!("{prefix}.layernorm2.weight"))?,
        })
    }
    fn ln(&self, w: &Tensor, x: &Tensor) -> crate::Result<Tensor> {
        RmsNorm::from_weight(w.clone()).forward(x)
    }
    pub fn forward(&self, x_q: &Tensor, x_kv: &Tensor) -> crate::Result<Tensor> {
        let attn_out = self.attn.forward(&self.ln(&self.ln_q_w, x_q)?, &self.ln(&self.ln_kv_w, x_kv)?)?;
        let x = (x_q + attn_out)?;
        let mlp_out = self.mlp.forward(&self.ln(&self.ln2_w, &x)?)?;
        Ok((x + mlp_out)?)
    }
}

pub struct InducedBlock {
    pub block1: CrossAttentionBlock,
    pub block2: CrossAttentionBlock,
    pub inducing_vectors: Tensor,
}

impl InducedBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, scaling: Option<SoftmaxScalingMlp>, n_ind: usize) -> crate::Result<Self> {
        Ok(Self {
            block1: CrossAttentionBlock::load(vb, &format!("{prefix}.cross_attn_block1"), emsize, num_heads, scaling)?,
            block2: CrossAttentionBlock::load(vb, &format!("{prefix}.cross_attn_block2"), emsize, num_heads, None)?,
            inducing_vectors: vb.get((n_ind, emsize), &format!("{prefix}.inducing_vectors"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, num_train: usize, cached_hidden: Option<&Tensor>) -> crate::Result<(Tensor, Tensor)> {
        let (bc, r) = (x.dim(0)?, x.dim(1)?);
        let e = x.dim(2)?;
        let hidden = if let Some(h) = cached_hidden {
            h.clone()
        } else {
            let n = num_train.min(r);
            let ind = self.inducing_vectors.unsqueeze(0)?.broadcast_as((bc, self.inducing_vectors.dim(0)?, e))?;
            let train = x.narrow(1, 0, n)?;
            self.block1.forward(&ind, &train)?
        };
        let out = self.block2.forward(&x, &hidden)?;
        Ok((out, hidden))
    }
}

pub struct FeatureDistributionEmbedder {
    pub layers: Vec<InducedBlock>,
}

impl FeatureDistributionEmbedder {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, num_layers: usize, n_ind: usize, scaling_hidden: usize) -> crate::Result<Self> {
        let head_dim = emsize / num_heads;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let scaling = Some(SoftmaxScalingMlp::load(
                vb,
                &format!("{prefix}.layers.{i}.cross_attn_block1.attn.softmax_scaling_layer"),
                num_heads,
                head_dim,
                scaling_hidden,
            )?);
            layers.push(InducedBlock::load(vb, &format!("{prefix}.layers.{i}"), emsize, num_heads, scaling, n_ind)?);
        }
        Ok(Self { layers })
    }

    /// x: (B, R, C, E) -> (B, R, C, E). cached_hidden: per-layer (B*C, n_ind, E).
    pub fn forward(&self, x: &Tensor, num_train: usize, cached_hidden: Option<&[Tensor]>) -> crate::Result<(Tensor, Vec<Tensor>)> {
        let (b, r, c, e) = (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?);
        let x_bcre = x.transpose(1, 2)?.contiguous()?;
        let mut cur = x_bcre.reshape((b * c, r, e))?;
        let mut hidden_out: Vec<Tensor> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let cached = cached_hidden.map(|h| &h[i]);
            let (out, hidden) = layer.forward(&cur, num_train, cached)?;
            if cached_hidden.is_none() {
                hidden_out.push(hidden);
            }
            cur = out;
        }
        let out_bcre = cur.reshape((b, c, r, e))?;
        Ok((out_bcre.transpose(1, 2)?.contiguous()?, hidden_out))
    }
}

pub struct AttentionRoPE {
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub out_w: Tensor,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AttentionRoPE {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize) -> crate::Result<Self> {
        let head_dim = emsize / num_heads;
        let hd = head_dim * num_heads;
        let g = |n: &str, s: (usize, usize)| vb.get(s, &format!("{prefix}.attention.{n}.weight"));
        Ok(Self {
            q_w: g("q_projection", (hd, emsize))?,
            k_w: g("k_projection", (hd, emsize))?,
            v_w: g("v_projection", (hd, emsize))?,
            out_w: vb.get((emsize, hd), &format!("{prefix}.attention.out_projection.weight"))?,
            q_norm: RmsNorm::load_dim(vb, &format!("{prefix}.attention.q_norm.weight"), head_dim)?,
            k_norm: RmsNorm::load_dim(vb, &format!("{prefix}.attention.k_norm.weight"), head_dim)?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> crate::Result<Tensor> {
        let (b, s) = (x.dim(0)?, x.dim(1)?);
        let h = self.num_heads;
        let d = self.head_dim;
        let q = linear_no_bias(x, &self.q_w)?.reshape((b, s, h, d))?;
        let k = linear_no_bias(x, &self.k_w)?.reshape((b, s, h, d))?;
        let v = linear_no_bias(x, &self.v_w)?.reshape((b, s, h, d))?;
        let q = rope.apply(&q.transpose(1, 2)?)?.transpose(1, 2)?;
        let k = rope.apply(&k.transpose(1, 2)?)?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let out = sdpa(&q, &k, &v)?;
        let out = out.reshape((b, s, h * d))?;
        Ok(linear_no_bias(&out, &self.out_w)?)
    }

    pub fn forward_cross(&self, query: &Tensor, context: &Tensor, rope: &RotaryEmbedding) -> crate::Result<Tensor> {
        let (b, qlen) = (query.dim(0)?, query.dim(1)?);
        let vlen = context.dim(1)?;
        let h = self.num_heads;
        let d = self.head_dim;
        let q = linear_no_bias(query, &self.q_w)?.reshape((b, qlen, h, d))?;
        let k = linear_no_bias(context, &self.k_w)?.reshape((b, vlen, h, d))?;
        let v = linear_no_bias(context, &self.v_w)?.reshape((b, vlen, h, d))?;
        let q = rope.apply(&q.transpose(1, 2)?)?.transpose(1, 2)?;
        let k = rope.apply(&k.transpose(1, 2)?)?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let out = sdpa(&q, &k, &v)?;
        let out = out.reshape((b, qlen, h * d))?;
        Ok(linear_no_bias(&out, &self.out_w)?)
    }
}

pub struct TransformerBlock {
    pub attention: AttentionRoPE,
    pub ln_w: Tensor,
    pub ln_mlp_w: Tensor,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize) -> crate::Result<Self> {
        let hidden = emsize * 2;
        Ok(Self {
            attention: AttentionRoPE::load(vb, prefix, emsize, num_heads)?,
            ln_w: vb.get(emsize, &format!("{prefix}.layernorm.weight"))?,
            ln_mlp_w: vb.get(emsize, &format!("{prefix}.layernorm_mlp.weight"))?,
            mlp: Mlp::load(vb, &format!("{prefix}.mlp"), emsize, hidden)?,
        })
    }
    fn ln(&self, w: &Tensor, x: &Tensor) -> crate::Result<Tensor> {
        RmsNorm::from_weight(w.clone()).forward(x)
    }
    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> crate::Result<Tensor> {
        let (b, r, s, e) = (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?);
        let xf = x.reshape((b * r, s, e))?;
        let delta = self.attention.forward(&self.ln(&self.ln_w, &xf)?, rope)?;
        let x = (xf + delta)?.reshape((b, r, s, e))?;
        let xf = x.reshape((b * r, s, e))?;
        let delta = self.mlp.forward(&self.ln(&self.ln_mlp_w, &xf)?)?;
        Ok(((xf + delta)?).reshape((b, r, s, e))?)
    }
    pub fn forward_cross(&self, query: &Tensor, context: &Tensor, rope: &RotaryEmbedding) -> crate::Result<Tensor> {
        let (b, r, q, e) = (query.dim(0)?, query.dim(1)?, query.dim(2)?, query.dim(3)?);
        let v = context.dim(2)?;
        let qf = query.reshape((b * r, q, e))?;
        let cf = context.reshape((b * r, v, e))?;
        let nq = self.ln(&self.ln_w, &qf)?;
        let nc = self.ln(&self.ln_w, &cf)?;
        let attn_out = self.attention.forward_cross(&nq, &nc, rope)?;
        let x = (qf + attn_out)?.reshape((b, r, q, e))?;
        let xf = x.reshape((b * r, q, e))?;
        let delta = self.mlp.forward(&self.ln(&self.ln_mlp_w, &xf)?)?;
        Ok(((xf + delta)?).reshape((b, r, q, e))?)
    }
}

pub struct ColumnAggregator {
    pub blocks: Vec<TransformerBlock>,
    pub rope: RotaryEmbedding,
    pub cls_tokens: Tensor,
    pub out_ln_w: Tensor,
    pub num_cls: usize,
}

impl ColumnAggregator {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, num_layers: usize, num_cls: usize, rope_base: f32, device: &candle_core::Device) -> crate::Result<Self> {
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(TransformerBlock::load(vb, &format!("{prefix}.blocks.{i}"), emsize, num_heads)?);
        }
        let head_dim = emsize / num_heads;
        Ok(Self {
            blocks,
            rope: RotaryEmbedding::new(head_dim, rope_base, device)?,
            cls_tokens: vb.get((num_cls, emsize), &format!("{prefix}.cls_tokens"))?,
            out_ln_w: vb.get(emsize, &format!("{prefix}.out_ln.weight"))?,
            num_cls,
        })
    }

    /// x: (B, R, C, E) -> (B, R, cls, E)
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let (b, r, _c, e) = (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?);
        let cls = self.cls_tokens.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, r, self.num_cls, e))?;
        let mut x = Tensor::cat(&[&cls, x], 2)?;
        for block in &self.blocks[..self.blocks.len() - 1] {
            x = block.forward(&x, &self.rope)?;
        }
        let last = &self.blocks[self.blocks.len() - 1];
        let cls_part = x.narrow(2, 0, self.num_cls)?;
        let out = last.forward_cross(&cls_part, &x, &self.rope)?;
        RmsNorm::from_weight(self.out_ln_w.clone()).forward(&out)
    }
}

pub struct IclAttention {
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub out_w: Tensor,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_kv_heads_test: Option<usize>,
    pub head_dim: usize,
    pub scaling: Option<SoftmaxScalingMlp>,
}

impl IclAttention {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, num_kv_heads: Option<usize>, num_kv_heads_test: Option<usize>, scaling: Option<SoftmaxScalingMlp>) -> crate::Result<Self> {
        let head_dim = emsize / num_heads;
        let q_hd = head_dim * num_heads;
        let kv_heads = num_kv_heads.unwrap_or(num_heads);
        let kv_hd = head_dim * kv_heads;
        Ok(Self {
            q_w: vb.get((q_hd, emsize), &format!("{prefix}.q_projection.weight"))?,
            k_w: vb.get((kv_hd, emsize), &format!("{prefix}.k_projection.weight"))?,
            v_w: vb.get((kv_hd, emsize), &format!("{prefix}.v_projection.weight"))?,
            out_w: vb.get((emsize, q_hd), &format!("{prefix}.out_projection.weight"))?,
            q_norm: RmsNorm::load_dim(vb, &format!("{prefix}.q_norm.weight"), head_dim)?,
            k_norm: RmsNorm::load_dim(vb, &format!("{prefix}.k_norm.weight"), head_dim)?,
            num_heads,
            num_kv_heads: kv_heads,
            num_kv_heads_test,
            head_dim,
            scaling,
        })
    }

    pub fn forward(&self, x: &Tensor, num_train: usize, cached_kv: Option<(&Tensor, &Tensor)>) -> crate::Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let b = x.dim(0)?;
        let r = x.dim(1)?;
        let h = self.num_heads;
        let d = self.head_dim;
        let q = self.q_norm.forward(&linear_no_bias(x, &self.q_w)?.reshape((b, r, h, d))?)?;

        if let Some((k, v)) = cached_kv {
            let out = crate::attention::sdpa_scaled(&q, k, v, self.scaling.as_ref())?;
            let out = out.reshape((b, r, h * d))?;
            return Ok((linear_no_bias(&out, &self.out_w)?, None));
        }

        let n = num_train.min(r);
        let x_train = x.narrow(1, 0, n)?;
        let kt = linear_no_bias(&x_train, &self.k_w)?.reshape((b, n, self.num_kv_heads, d))?;
        let vt = linear_no_bias(&x_train, &self.v_w)?.reshape((b, n, self.num_kv_heads, d))?;
        let kt = self.k_norm.forward(&kt)?;

        let store_k = if let Some(ht) = self.num_kv_heads_test {
            kt.narrow(2, 0, ht)?.contiguous()?
        } else {
            kt.clone()
        };
        let store_v = if let Some(ht) = self.num_kv_heads_test {
            vt.narrow(2, 0, ht)?.contiguous()?
        } else {
            vt.clone()
        };

        let out = if self.num_kv_heads_test.is_some() && n < r {
            let ht = self.num_kv_heads_test.unwrap();
            let out_train = crate::attention::sdpa_scaled(&q.narrow(1, 0, n)?, &kt, &vt, self.scaling.as_ref())?;
            let out_test = crate::attention::sdpa_scaled(&q.narrow(1, n, r - n)?, &kt.narrow(2, 0, ht)?, &vt.narrow(2, 0, ht)?, self.scaling.as_ref())?;
            Tensor::cat(&[&out_train, &out_test], 1)?
        } else {
            crate::attention::sdpa_scaled(&q, &kt, &vt, self.scaling.as_ref())?
        };
        let out = out.reshape((b, r, h * d))?;
        Ok((linear_no_bias(&out, &self.out_w)?, Some((store_k, store_v))))
    }
}

pub struct IclBlock {
    pub attn: IclAttention,
    pub ln_w: Tensor,
    pub ln_mlp_w: Tensor,
    pub mlp: Mlp,
}

impl IclBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize, num_heads: usize, num_kv_heads: Option<usize>, num_kv_heads_test: Option<usize>, scaling: Option<SoftmaxScalingMlp>) -> crate::Result<Self> {
        let hidden = emsize * 2;
        Ok(Self {
            attn: IclAttention::load(vb, &format!("{prefix}.icl_attention"), emsize, num_heads, num_kv_heads, num_kv_heads_test, scaling)?,
            ln_w: vb.get(emsize, &format!("{prefix}.layernorm.weight"))?,
            ln_mlp_w: vb.get(emsize, &format!("{prefix}.layernorm_mlp.weight"))?,
            mlp: Mlp::load(vb, &format!("{prefix}.mlp"), emsize, hidden)?,
        })
    }
    fn ln(&self, w: &Tensor, x: &Tensor) -> crate::Result<Tensor> {
        RmsNorm::from_weight(w.clone()).forward(x)
    }
    pub fn forward(&self, x: &Tensor, num_train: usize, cached: Option<(&Tensor, &Tensor)>) -> crate::Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (out, kv) = self.attn.forward(&self.ln(&self.ln_w, x)?, num_train, cached)?;
        let x = (x + out)?;
        let mlp_out = self.mlp.forward(&self.ln(&self.ln_mlp_w, &x)?)?;
        Ok(((x + mlp_out)?, kv))
    }
}

pub struct PreHeadMlp {
    pub w: Tensor,
    pub mlp: Mlp,
}

impl PreHeadMlp {
    pub fn load(vb: &VarBuilder, prefix: &str, emsize: usize) -> crate::Result<Self> {
        let hidden = emsize * 2;
        Ok(Self {
            w: vb.get(emsize, &format!("{prefix}.norm.weight"))?,
            mlp: Mlp::load(vb, &format!("{prefix}.mlp"), emsize, hidden)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let d = self.mlp.forward(&RmsNorm::from_weight(self.w.clone()).forward(x)?)?;
        Ok((x + d)?)
    }
}

/// Manual embedding lookup (candle_nn has no ops::embedding).
pub fn embedding_lookup(ids: &[u32], w: &Tensor) -> crate::Result<Tensor> {
    let (vocab, dim) = (w.dim(0)?, w.dim(1)?);
    let mut data = vec![0.0f32; ids.len() * dim];
    let wf = w.to_vec2::<f32>()?;
    for (i, &id) in ids.iter().enumerate() {
        let row = &wf[id as usize % vocab];
        data[i * dim..(i + 1) * dim].copy_from_slice(row);
    }
    Ok(Tensor::from_vec(data, (ids.len(), dim), w.device())?)
}
