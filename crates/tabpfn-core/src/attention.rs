//! Tiled (query-chunked) scaled dot-product attention. Memory-safe for WASM.

use candle_core::Tensor;

use crate::layers::SoftmaxScalingMlp;

pub const Q_CHUNK: usize = 128;

/// q: (B, S, H, D), k/v: (B, J, Hkv, D). GQA: head h uses kv head h/(H/Hkv).
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> crate::Result<Tensor> {
    let (b, s, h, d) = (q.dim(0)?, q.dim(1)?, q.dim(2)?, q.dim(3)?);
    let hkv = k.dim(2)?;
    let scale = (d as f32).sqrt();

    let mut outs: Vec<Tensor> = Vec::with_capacity(h);
    for head in 0..h {
        let kv_head = if hkv == h { head } else { head / (h / hkv) };
        let qh = q.narrow(2, head, 1)?.squeeze(2)?; // (B, S, D)
        let kh = k.narrow(2, kv_head, 1)?.squeeze(2)?; // (B, J, D)
        let vh = v.narrow(2, kv_head, 1)?.squeeze(2)?;
        let mut head_outs: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < s {
            let chunk = Q_CHUNK.min(s - start);
            let qc = qh.narrow(1, start, chunk)?; // (B, chunk, D)
            let scores = qc.matmul(&kh.transpose(1, 2)?)?; // (B, chunk, J)
            let scores = (scores / (scale as f64))?;
            let attn = candle_nn::ops::softmax(&scores, 2)?;
            head_outs.push(attn.matmul(&vh)?);
            start += chunk;
        }
        let head_out = if head_outs.len() == 1 {
            head_outs.into_iter().next().unwrap()
        } else {
            Tensor::cat(&head_outs.iter().collect::<Vec<_>>(), 1)?
        };
        outs.push(head_out);
    }
    let mut stacked = outs[0].unsqueeze(2)?; // (B, S, 1, D)
    for o in &outs[1..] {
        stacked = Tensor::cat(&[&stacked, &o.unsqueeze(2)?], 2)?;
    }
    Ok(stacked)
}

pub fn sdpa_scaled(q: &Tensor, k: &Tensor, v: &Tensor, scaling: Option<&SoftmaxScalingMlp>) -> crate::Result<Tensor> {
    if let Some(mlp) = scaling {
        let n = k.dim(1)?;
        let q = mlp.forward(q, n)?;
        sdpa(&q, k, v)
    } else {
        sdpa(q, k, v)
    }
}
