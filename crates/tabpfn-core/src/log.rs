//! Manual LayerNorm (avoids candle_nn::LayerNorm Device/Module quirks).
//! y = (x - mean) / sqrt(var + eps) * w + b, over last dim, f32.

use candle_core::{Tensor, D};

pub fn layer_norm_forward(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> crate::Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps)?.sqrt()?;
    let normed = centered.broadcast_div(&denom)?;
    Ok(normed.broadcast_mul(w)?.broadcast_add(b)?)
}
