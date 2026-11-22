use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::attention::sdpa;
use crate::cache::{KvEntry, TabPFNV3p5Cache};
use crate::config::{TabPFNV3p5Config, TaskType};
use crate::layers::*;
use crate::stages::*;
use crate::Error;

pub struct XEmbed {
    pub fourier: FourierEmbedder,
    pub metadata_w: Tensor,
    pub ln_w: Tensor,
    pub ln_b: Tensor,
    pub group_size: usize,
    pub ecdf_num_frequencies: usize,
}

impl XEmbed {
    pub fn load(vb: &VarBuilder, config: &TabPFNV3p5Config) -> crate::Result<Self> {
        let g = config.feature_group_size;
        let e = config.embed_dim;
        let f = config.fourier_encoding_num_frequencies;
        let k = config.cell_ecdf_num_frequencies;
        let meta_width = g * 2 + g * 2 * k;
        Ok(Self {
            fourier: FourierEmbedder::load(vb, "x_embed.fourier", g, f, e)?,
            metadata_w: vb.get((e, meta_width), "x_embed.metadata_linear.weight")?,
            ln_w: vb.get(e, "x_embed.layernorm.weight")?,
            ln_b: vb.get(e, "x_embed.layernorm.bias")?,
            group_size: g,
            ecdf_num_frequencies: k,
        })
    }

    pub fn forward(&self, x_grouped: &Tensor) -> crate::Result<Tensor> {
        let (b, r, c) = (x_grouped.dim(0)?, x_grouped.dim(1)?, x_grouped.dim(2)?);
        let chunk = 2048usize;
        let mut parts = Vec::new();
        let mut start = 0;
        while start < r {
            let len = chunk.min(r - start);
            let xc = x_grouped.narrow(1, start, len)?;
            parts.push(self.embed_one(&xc, b, len, c)?);
            start += len;
        }
        if parts.len() == 1 {
            Ok(parts.into_iter().next().unwrap())
        } else {
            Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?)
        }
    }

    fn embed_one(&self, xc: &Tensor, b: usize, r: usize, c: usize) -> crate::Result<Tensor> {
        let g = self.group_size;
        let k = self.ecdf_num_frequencies;
        let vals = xc.narrow(3, 0, g)?;
        let fourier = self.fourier.forward(&vals)?;
        let meta_head = xc.narrow(3, 0, 2 * g)?;
        let ranks = xc.narrow(3, 2 * g, g)?;
        let n = b * r * c;
        let ranks_f = ranks.reshape((n, g))?;
        // _ecdf_fourier_features: phase = u * pi*k for k=1..K, then [sin(phase), cos(phase)]
        let mut sin_chunks = Vec::with_capacity(k);
        let mut cos_chunks = Vec::with_capacity(k);
        for ki in 1..=k {
            let phase = ranks_f.broadcast_mul(&Tensor::new((std::f32::consts::PI as f64 * ki as f64) as f32, ranks_f.device())?)?;
            sin_chunks.push(phase.sin()?);
            cos_chunks.push(phase.cos()?);
        }
        // _ecdf_fourier_features(u): phase = u.unsqueeze(-1)*pi*k -> (..., G, K); [sin, cos] cat last -> (..., G, 2K); flatten(-2).
        // For a single cell u=(G,), phase (G,K), lifted (G,2K) flattened row-major: [(g0: sin k0..kK-1, cos k0..), g1: ...].
        let mut lifted_data: Vec<f32> = Vec::with_capacity(n * 2 * k * g);
        let mut sin_blocks: Vec<Vec<f32>> = Vec::with_capacity(k);
        for t in sin_chunks.iter() {
            sin_blocks.push(t.flatten_all()?.to_vec1::<f32>()?);
        }
        let mut cos_blocks: Vec<Vec<f32>> = Vec::with_capacity(k);
        for t in cos_chunks.iter() {
            cos_blocks.push(t.flatten_all()?.to_vec1::<f32>()?);
        }
        // sin_blocks[ki] is (n*g,) with layout [n0: g0..g(G-1), n1: ...]. For element (ni, gi, ki): sin_blocks[ki][ni*g+gi].
        // Per (n, g): [sin k0..k(K-1), cos k0..k(K-1)] — interleaved per group.
        for ni in 0..n {
            for gi in 0..g {
                for ki in 0..k {
                    lifted_data.push(sin_blocks[ki][ni * g + gi]);
                }
                for ki in 0..k {
                    lifted_data.push(cos_blocks[ki][ni * g + gi]);
                }
            }
        }
        let lifted = Tensor::from_vec(lifted_data, (n, 2 * k * g), ranks_f.device())?;
        let head = meta_head.reshape((n, 2 * g))?;
        let meta = Tensor::cat(&[&head, &lifted], 1)?.reshape((b, r, c, 2 * g + 2 * k * g))?;
        let metadata = crate::layers::linear_no_bias(&meta, &self.metadata_w)?;
        let summed = (fourier + metadata)?;
        crate::log::layer_norm_forward(&summed, &self.ln_w, &self.ln_b, 1e-5)
    }
}

pub struct YEncoder {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub is_regression: bool,
    pub ln_w: Tensor,
    pub ln_b: Tensor,
}

impl YEncoder {
    pub fn load_embedding(vb: &VarBuilder, prefix: &str, classes: usize, dim: usize, ln_prefix: &str) -> crate::Result<Self> {
        Ok(Self {
            w: vb.get((classes, dim), &format!("{prefix}.embedding.weight"))?,
            b: None,
            is_regression: false,
            ln_w: vb.get(dim, &format!("{ln_prefix}.weight"))?,
            ln_b: vb.get(dim, &format!("{ln_prefix}.bias"))?,
        })
    }
    pub fn load_linear(vb: &VarBuilder, prefix: &str, dim: usize, ln_prefix: &str) -> crate::Result<Self> {
        Ok(Self {
            w: vb.get((dim, 1), &format!("{prefix}.weight"))?,
            b: Some(vb.get(dim, &format!("{prefix}.bias"))?),
            is_regression: true,
            ln_w: vb.get(dim, &format!("{ln_prefix}.weight"))?,
            ln_b: vb.get(dim, &format!("{ln_prefix}.bias"))?,
        })
    }

    pub fn embed(&self, y_bn: &Tensor) -> crate::Result<Tensor> {
        let (b, n) = (y_bn.dim(0)?, y_bn.dim(1)?);
        let out = if self.is_regression {
            let y_col = y_bn.reshape((b * n, 1))?;
            crate::layers::linear(&y_col, &self.w, self.b.as_ref().unwrap())?.reshape((b, n, self.w.dim(0)?))?
        } else {
            let flat = y_bn.reshape((b * n,))?.to_vec1::<f32>()?;
            let ids: Vec<u32> = flat.iter().map(|&v| (v as u32)).collect();
            embedding_lookup(&ids, &self.w)?.reshape((b, n, self.w.dim(1)?))?
        };
        crate::log::layer_norm_forward(&out, &self.ln_w, &self.ln_b, 1e-5)
    }
}

pub struct ManyClassDecoder {
    pub q_w: Tensor,
    pub q_b: Tensor,
    pub k_w: Tensor,
    pub k_b: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scaling: Option<SoftmaxScalingMlp>,
    pub max_num_classes: usize,
}

impl ManyClassDecoder {
    pub fn load(vb: &VarBuilder, config: &TabPFNV3p5Config) -> crate::Result<Self> {
        let input = config.icl_emsize();
        let att = config.decoder_head_dim * config.decoder_num_heads;
        let scaling = if config.decoder_use_softmax_scaling {
            Some(SoftmaxScalingMlp::load(vb, "heads.many_class_decoder.softmax_scaling_layer", config.decoder_num_heads, config.decoder_head_dim, config.softmax_scaling_mlp_hidden_dim)?)
        } else {
            None
        };
        Ok(Self {
            q_w: vb.get((att, input), "heads.many_class_decoder.q_projection.weight")?,
            q_b: vb.get(att, "heads.many_class_decoder.q_projection.bias")?,
            k_w: vb.get((att, input), "heads.many_class_decoder.k_projection.weight")?,
            k_b: vb.get(att, "heads.many_class_decoder.k_projection.bias")?,
            num_heads: config.decoder_num_heads,
            head_dim: config.decoder_head_dim,
            scaling,
            max_num_classes: config.max_num_classes,
        })
    }

    pub fn project_keys(&self, train_emb: &Tensor) -> crate::Result<Tensor> {
        let (b, n, _) = (train_emb.dim(0)?, train_emb.dim(1)?, train_emb.dim(2)?);
        let k = crate::layers::linear(train_emb, &self.k_w, &self.k_b)?;
        Ok(k.reshape((b, n, self.num_heads, self.head_dim))?)
    }

    pub fn forward(&self, keys: &Tensor, test_emb: &Tensor, targets: &Tensor, num_present_classes: usize) -> crate::Result<Tensor> {
        let (b, m, _) = (test_emb.dim(0)?, test_emb.dim(1)?, test_emb.dim(2)?);
        let n = keys.dim(1)?;
        if m == 0 {
            return Ok(Tensor::zeros((0, b, self.max_num_classes), candle_core::DType::F32, test_emb.device())?);
        }
        let q_raw = crate::layers::linear(test_emb, &self.q_w, &self.q_b)?;
        let mut q = q_raw.reshape((b, m, self.num_heads, self.head_dim))?;
        if let Some(s) = &self.scaling {
            q = s.forward(&q, n)?;
        }
        let tgt = targets.to_vec2::<f32>()?;
        let mut tgt_ids = vec![vec![-1i64; n]; b];
        for bi in 0..b {
            for ni in 0..n {
                let v = tgt[bi][ni];
                if v.is_finite() {
                    tgt_ids[bi][ni] = v as i64;
                }
            }
        }
        let c = num_present_classes;
        let d = self.head_dim;
        let h = self.num_heads;
        let mut chunks: Vec<Tensor> = Vec::new();
        let mut first = 0;
        while first < c {
            let width = d.min(c - first);
            let mut v_data = vec![0.0f32; b * n * width];
            for bi in 0..b {
                for ni in 0..n {
                    let t = tgt_ids[bi][ni] - first as i64;
                    if t >= 0 && (t as usize) < width {
                        v_data[(bi * n + ni) * width + t as usize] = 1.0;
                    }
                }
            }
            let v = Tensor::from_vec(v_data, (b, n, width), keys.device())?
                .unsqueeze(2)?
                .broadcast_as((b, n, h, width))?;
            chunks.push(sdpa(&q, keys, &v)?);
            first += d;
        }
        let cat = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 3)?;
        let sliced = cat.narrow(3, 0, c)?;
        let mean_heads = sliced.mean(2)?;
        let padded = if c < self.max_num_classes {
            let zeros = Tensor::zeros((b, m, self.max_num_classes - c), candle_core::DType::F32, mean_heads.device())?;
            Tensor::cat(&[&mean_heads, &zeros], 2)?
        } else {
            mean_heads
        };
        let out = padded.transpose(0, 1)?;
        let clamped = out.clamp(1e-5f64, f64::INFINITY)?;
        Ok((clamped + 3e-5f64)?.log()?)
    }
}

pub struct MultiTaskHeads {
    pub decoder: ManyClassDecoder,
    pub output_w: Tensor,
    pub output_b: Tensor,
    pub mlp_cls: PreHeadMlp,
    pub mlp_reg: PreHeadMlp,
}

impl MultiTaskHeads {
    pub fn load(vb: &VarBuilder, config: &TabPFNV3p5Config) -> crate::Result<Self> {
        let input = config.icl_emsize();
        Ok(Self {
            decoder: ManyClassDecoder::load(vb, config)?,
            output_w: vb.get((config.num_buckets, input), "heads.output_projection.weight")?,
            output_b: vb.get(config.num_buckets, "heads.output_projection.bias")?,
            mlp_cls: PreHeadMlp::load(vb, "heads.mlp_classification", input)?,
            mlp_reg: PreHeadMlp::load(vb, "heads.mlp_regression", input)?,
        })
    }
    pub fn project_keys(&self, train_emb: &Tensor) -> crate::Result<Tensor> {
        self.decoder.project_keys(&self.mlp_cls.forward(train_emb)?)
    }
}

pub struct TabPFNV3p5 {
    pub config: TabPFNV3p5Config,
    pub x_embed: XEmbed,
    pub col_y_multiclass: YEncoder,
    pub col_y_regression: YEncoder,
    pub dist_embed: FeatureDistributionEmbedder,
    pub col_agg: ColumnAggregator,
    pub icl_y_multiclass: YEncoder,
    pub icl_y_regression: YEncoder,
    pub icl_blocks: Vec<IclBlock>,
    pub out_ln_w: Tensor,
    pub heads: MultiTaskHeads,
    pub device: Device,
}

impl TabPFNV3p5 {
    pub fn from_safetensors(bytes: &[u8], config: TabPFNV3p5Config, device: &Device) -> crate::Result<Self> {
        config.validate()?;
        let tensors = safetensors::SafeTensors::deserialize(bytes).map_err(|e| Error::Safetensors(e.to_string()))?;
        let vb = candle_nn::VarBuilder::from_slice_safetensors(bytes, candle_core::DType::F32, device)
            .map_err(|e| Error::Safetensors(e.to_string()))?;
        Self::from_var_builder(vb, config, device)
    }

    pub fn from_var_builder(vb: candle_nn::VarBuilder, config: TabPFNV3p5Config, device: &Device) -> crate::Result<Self> {
        let e = config.embed_dim;
        let icl_emsize = config.icl_emsize();
        let x_embed = XEmbed::load(&vb, &config)?;
        let col_y_multiclass = YEncoder::load_embedding(&vb, "col_y_encoder.multiclass", config.max_num_classes, e, "col_y_layernorm")?;
        let col_y_regression = YEncoder::load_linear(&vb, "col_y_encoder.regression", e, "col_y_layernorm")?;
        let dist_embed = FeatureDistributionEmbedder::load(&vb, "feature_distribution_embedder", e, config.dist_embed_num_heads, config.dist_embed_num_blocks, config.dist_embed_num_inducing_points, config.softmax_scaling_mlp_hidden_dim)?;
        let col_agg = ColumnAggregator::load(&vb, "column_aggregator", e, config.feat_agg_num_heads, config.feat_agg_num_blocks, config.feat_agg_num_cls_tokens, config.feat_agg_rope_base, device)?;
        let icl_y_multiclass = YEncoder::load_embedding(&vb, "icl_y_encoder.multiclass", config.max_num_classes, icl_emsize, "icl_y_layernorm")?;
        let icl_y_regression = YEncoder::load_linear(&vb, "icl_y_encoder.regression", icl_emsize, "icl_y_layernorm")?;
        let mut icl_blocks = Vec::with_capacity(config.nlayers);
        for i in 0..config.nlayers {
            let scaling = Some(SoftmaxScalingMlp::load(&vb, &format!("icl_blocks.{i}.icl_attention.softmax_scaling_layer"), config.icl_num_heads, config.icl_head_dim(), config.softmax_scaling_mlp_hidden_dim)?);
            icl_blocks.push(IclBlock::load(&vb, &format!("icl_blocks.{i}"), icl_emsize, config.icl_num_heads, config.icl_num_kv_heads, config.icl_num_kv_heads_test, scaling)?);
        }
        Ok(Self {
            x_embed,
            col_y_multiclass,
            col_y_regression,
            dist_embed,
            col_agg,
            icl_y_multiclass,
            icl_y_regression,
            icl_blocks,
            out_ln_w: vb.get(icl_emsize, "output_norm.weight")?,
            heads: MultiTaskHeads::load(&vb, &config)?,
            config,
            device: device.clone(),
        })
    }

    fn y_encoder(&self, task: TaskType, col: bool) -> &YEncoder {
        match (task, col) {
            (TaskType::Multiclass, true) => &self.col_y_multiclass,
            (TaskType::Regression, true) => &self.col_y_regression,
            (TaskType::Multiclass, false) => &self.icl_y_multiclass,
            (TaskType::Regression, false) => &self.icl_y_regression,
        }
    }

    fn prepare_y_col(&self, y: &[f32], task: TaskType) -> Vec<f32> {
        let mut out = crate::preprocess::impute_targets(y, y.len());
        if task == TaskType::Multiclass {
            for (o, &orig) in out.iter_mut().zip(y.iter()) {
                if !orig.is_finite() {
                    *o = o.ceil();
                }
            }
        }
        out
    }

    pub fn forward(
        &self,
        x_brc: &Tensor,
        y_train: &[f32],
        task: TaskType,
        cache: Option<&TabPFNV3p5Cache>,
        x_is_test_only: bool,
        num_present_classes: Option<usize>,
    ) -> crate::Result<Tensor> {
        let (b, r, _c) = (x_brc.dim(0)?, x_brc.dim(1)?, x_brc.dim(2)?);
        let n = y_train.len();
        let num_present_classes = num_present_classes.unwrap_or_else(|| {
            if task == TaskType::Multiclass {
                (y_train.iter().fold(0.0f32, |m, &v| if v.is_finite() { m.max(v) } else { m }) as usize + 1).max(1)
            } else {
                0
            }
        });
        let (stages_out, _, _, _) = self.stages_0_to_2_impl(x_brc, y_train, task, cache, x_is_test_only)?;
        let x = stages_out.reshape((b, r, self.config.icl_emsize()))?;

        let running_from_cache = cache.is_some_and(|c| !c.is_empty());
        let x = if running_from_cache {
            let c = cache.unwrap();
            let mut x = x;
            for (i, block) in self.icl_blocks.iter().enumerate() {
                let (out, _) = block.forward(&x, 0, Some((&c.kv[i].key, &c.kv[i].value)))?;
                x = out;
            }
            x
        } else {
            let mut x = x;
            if n > 0 {
                let y_prep = self.prepare_y_col(y_train, task);
                let y_bn = Tensor::from_vec(y_prep, (b, n), &self.device)?;
                let y_emb = self.y_encoder(task, false).embed(&y_bn)?;
                let train = (x.narrow(1, 0, n)? + y_emb)?;
                let tail = x.narrow(1, n, r - n)?;
                x = Tensor::cat(&[&train, &tail], 1)?;
            }
            for block in self.icl_blocks.iter() {
                let (out, _) = block.forward(&x, n, None)?;
                x = out;
            }
            x
        };
        let x = RmsNorm::from_weight(self.out_ln_w.clone()).forward(&x)?;
        let (test_emb, train_keys) = if running_from_cache {
            (x.clone(), cache.unwrap().decoder_keys.clone())
        } else {
            let test_emb = x.narrow(1, n, r - n)?;
            let keys = if task == TaskType::Multiclass {
                Some(self.heads.project_keys(&x.narrow(1, 0, n)?)?)
            } else {
                None
            };
            (test_emb, keys)
        };
        let y_bn = Tensor::from_vec(y_train.to_vec(), (b, n), &self.device)?;
        let out = self.heads_forward(train_keys.as_ref(), &test_emb, &y_bn, task, num_present_classes)?;
        // nan_to_num(0.0)
        let mut flat = out.to_vec3::<f32>()?;
        for mb in flat.iter_mut() {
            for row in mb.iter_mut() {
                for v in row.iter_mut() {
                    if v.is_nan() {
                        *v = 0.0;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(flat.iter().flat_map(|m| m.iter().flat_map(|r| r.iter().cloned())).collect::<Vec<_>>(), out.dims(), &self.device)?)
    }

    fn heads_forward(&self, train_keys: Option<&Tensor>, test_emb: &Tensor, y_bn: &Tensor, task: TaskType, num_present_classes: usize) -> crate::Result<Tensor> {
        match task {
            TaskType::Regression => {
                let emb = self.heads.mlp_reg.forward(test_emb)?;
                let out = crate::layers::linear(&emb, &self.heads.output_w, &self.heads.output_b)?;
                Ok(out.transpose(0, 1)?)
            }
            TaskType::Multiclass => {
                let emb = self.heads.mlp_cls.forward(test_emb)?;
                let keys = train_keys.expect("multiclass head needs decoder keys");
                self.heads.decoder.forward(keys, &emb, y_bn, num_present_classes)
            }
        }
    }

    fn stages_0_to_2_impl(
        &self,
        x_brc: &Tensor,
        y_train: &[f32],
        task: TaskType,
        cache: Option<&TabPFNV3p5Cache>,
        x_is_test_only: bool,
    ) -> crate::Result<(Tensor, crate::preprocess::ScalerCache, crate::preprocess::EcdfContext, Vec<Tensor>)> {
        let trace = std::env::var("TABPFN_STAGE").ok();
        let trace_enabled = trace.is_some();
        let dump = |name: &str, t: &Tensor| -> crate::Result<()> {
            if trace.as_deref() == Some(name) {
                let v = t.flatten_all()?.to_vec1::<f32>()?;
                let s: f32 = v.iter().map(|x| x.abs()).sum();
                eprintln!("{name}[:8]={:?} sum={s}", &v[..8.min(v.len())]);
            }
            Ok(())
        };
        let _ = trace_enabled;
        let (b, r_in, c) = (x_brc.dim(0)?, x_brc.dim(1)?, x_brc.dim(2)?);
        let n = y_train.len();
        let x_vec = x_brc.to_vec3::<f32>()?;
        let flat: Vec<f32> = x_vec.iter().flat_map(|v| v.iter().flat_map(|w| w.iter().copied())).collect();

        let running_from_cache = cache.is_some_and(|c| !c.is_empty());
        let rows_r = if running_from_cache {
            if x_is_test_only { r_in } else { r_in - n }
        } else {
            r_in
        };

        let nan_ind = crate::preprocess::nan_inf_indicator(&flat);
        let from_cache = cache.filter(|c| !c.is_empty()).map(|c| crate::preprocess::ScalerCache { mean: c.scaler_mean.clone(), std: c.scaler_std.clone() });
        let (means, scaler) = if let Some(s) = &from_cache {
            (s.mean.clone(), (*s).clone())
        } else {
            let mut sum = vec![0.0f32; b * c];
            let mut cnt = vec![0.0f32; b * c];
            for bi in 0..b {
                for ri in 0..n {
                    for ci in 0..c {
                        let v = flat[(bi * r_in + ri) * c + ci];
                        if v.is_finite() {
                            sum[bi * c + ci] += v;
                            cnt[bi * c + ci] += 1.0;
                        }
                    }
                }
            }
            let train_means: Vec<f32> = sum.iter().zip(cnt.iter()).map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 }).collect();
            let (imputed_tmp, _) = crate::preprocess::impute_with_means(&flat, &train_means, r_in, b, c);
            let scaler = crate::preprocess::fit_scaler(&imputed_tmp, n, b, c);
            (train_means, scaler)
        };

        let (mut imputed, _) = crate::preprocess::impute_with_means(&flat, &means, r_in, b, c);
        if from_cache.is_none() {
            for i in 0..imputed.len() {
                if !flat[i].is_finite() {
                    imputed[i] = scaler.mean[i % (b * c)];
                }
            }
        }

        let ecdf_ctx = if let Some(cc) = cache.filter(|c| !c.is_empty()) {
            cc.ecdf_context.clone()
        } else {
            let mut train_bnc = vec![0.0f32; b * n * c];
            for bi in 0..b {
                for ri in 0..n {
                    for ci in 0..c {
                        train_bnc[(bi * n + ri) * c + ci] = imputed[(bi * r_in + ri) * c + ci];
                    }
                }
            }
            crate::preprocess::build_ecdf_context(&train_bnc, b, n, c, n, self.config.cell_ecdf_num_buckets)
        };

        let ecdf = crate::preprocess::in_context_ecdf(&imputed, &ecdf_ctx, b, rows_r, c);
        let scaled_all = crate::preprocess::transform_scaler(&imputed, &scaler, b, c);
        let slice = |buf: &Vec<f32>| -> Vec<f32> {
            if running_from_cache && !x_is_test_only {
                let mut out = vec![0.0f32; b * rows_r * c];
                for bi in 0..b {
                    for ri in 0..rows_r {
                        for ci in 0..c {
                            out[(bi * rows_r + ri) * c + ci] = buf[(bi * r_in + n + ri) * c + ci];
                        }
                    }
                }
                out
            } else {
                buf.clone()
            }
        };
        let scaled = slice(&scaled_all);
        let nan_ind_s = slice(&nan_ind);
        let ecdf_s = slice(&ecdf);
        let grouped = crate::preprocess::group_features(&scaled, &nan_ind_s, &ecdf_s, b, rows_r, c, self.config.feature_group_size);
        let grouped_t = Tensor::from_vec(grouped.clone(), (b, rows_r, c, 3 * self.config.feature_group_size), &self.device)?;
        let mut x_emb = self.x_embed.forward(&grouped_t)?;
        let dump2 = |name: &str, t: &Tensor| -> crate::Result<()> {
            if std::env::var("TABPFN_STAGE").ok().as_deref() == Some(name) {
                let v = t.flatten_all()?.to_vec1::<f32>()?;
                let s: f32 = v.iter().map(|x| x.abs()).sum();
                eprintln!("{name}[:8]={:?} sum={s}", &v[..8.min(v.len())]);
            }
            Ok(())
        };
        dump2("x_embed", &x_emb)?;

        if !running_from_cache && n > 0 {
            let y_prep = self.prepare_y_col(y_train, task);
            let y_bn = Tensor::from_vec(y_prep, (b, n), &self.device)?;
            let emb = self.y_encoder(task, true).embed(&y_bn)?;
            let train_part = x_emb.narrow(1, 0, n)?;
            let train_dims = train_part.dims().to_vec();
            let new_train = (train_part + emb.unsqueeze(2)?.broadcast_as(train_dims.as_slice())?)?;
            x_emb = if rows_r > n {
                Tensor::cat(&[&new_train, &x_emb.narrow(1, n, rows_r - n)?], 1)?
            } else {
                new_train
            };
        }

        let cached_hidden: Option<Vec<Tensor>> = cache.filter(|c| !c.is_empty()).map(|c| c.inducing_hidden.clone());
        let num_train_for_dist = if running_from_cache { 0 } else { n };
        let (x_dist, new_hidden) = self.dist_embed.forward(&x_emb, num_train_for_dist, cached_hidden.as_deref())?;
        dump("x_dist", &x_dist)?;
        let inducing_hidden = if new_hidden.is_empty() { cached_hidden.clone().unwrap_or_default() } else { new_hidden };
        let x_agg = self.col_agg.forward(&x_dist)?;
        dump("x_agg", &x_agg)?;
        Ok((x_agg, scaler, ecdf_ctx, inducing_hidden))
    }

    pub fn fit(&self, x_train: &Tensor, y_train: &[f32], task: TaskType) -> crate::Result<TabPFNV3p5Cache> {
        let (b, n, c) = (x_train.dim(0)?, x_train.dim(1)?, x_train.dim(2)?);
        let (stages_out, scaler, ecdf_ctx, inducing_hidden) = self.stages_0_to_2_impl(x_train, y_train, task, None, false)?;
        let mut x = stages_out.reshape((b, n, self.config.icl_emsize()))?;
        let y_prep = self.prepare_y_col(y_train, task);
        let y_bn = Tensor::from_vec(y_prep, (b, n), &self.device)?;
        x = (x + self.y_encoder(task, false).embed(&y_bn)?)?;
        let mut kv = Vec::with_capacity(self.icl_blocks.len());
        for block in self.icl_blocks.iter() {
            let (out, entry) = block.forward(&x, n, None)?;
            x = out;
            let (k, v) = entry.expect("fit must return kv");
            kv.push(KvEntry { key: k, value: v });
        }
        let x = RmsNorm::from_weight(self.out_ln_w.clone()).forward(&x)?;
        let decoder_keys = if task == TaskType::Multiclass {
            Some(self.heads.project_keys(&x)?)
        } else {
            None
        };
        Ok(TabPFNV3p5Cache { kv, decoder_keys, train_shape: (b, n), scaler_mean: scaler.mean, scaler_std: scaler.std, ecdf_context: ecdf_ctx, inducing_hidden, num_train: n, n_cols: c })
    }

    pub fn predict_with_y(&self, cache: &TabPFNV3p5Cache, x_test: &Tensor, y_train: &[f32], task: TaskType) -> crate::Result<Vec<f32>> {
        let (b, m, c) = (x_test.dim(0)?, x_test.dim(1)?, x_test.dim(2)?);
        // Guard rails: a cache is only valid for the exact training set it was
        // built from (same task, same column count, same train targets).
        if c != cache.n_cols {
            return Err(Error::Shape(format!(
                "cache was fitted on {} columns but predict got {}; re-run fit() on this data",
                cache.n_cols, c
            )));
        }
        if b != cache.train_shape.0 {
            return Err(Error::Shape(format!(
                "cache batch {} != predict batch {b}",
                cache.train_shape.0
            )));
        }
        if y_train.len() != cache.num_train {
            return Err(Error::Shape(format!(
                "cache was fitted on {} train rows but {} targets were passed",
                cache.num_train,
                y_train.len()
            )));
        }
        if cache.kv.len() != self.icl_blocks.len() {
            return Err(Error::Shape(format!(
                "cache has {} ICL layers, model has {}",
                cache.kv.len(),
                self.icl_blocks.len()
            )));
        }
        if task == TaskType::Multiclass && cache.decoder_keys.is_none() {
            return Err(Error::Shape(
                "multiclass predict needs decoder keys; this cache was fitted for regression".into(),
            ));
        }
        let stages_out = self.stages_0_to_2_impl(x_test, y_train, task, Some(cache), true)?.0;
        let mut x = stages_out.reshape((b, m, self.config.icl_emsize()))?;
        for (i, block) in self.icl_blocks.iter().enumerate() {
            let (out, _) = block.forward(&x, 0, Some((&cache.kv[i].key, &cache.kv[i].value)))?;
            x = out;
        }
        let x = RmsNorm::from_weight(self.out_ln_w.clone()).forward(&x)?;
        let y_bn = Tensor::from_vec(y_train.to_vec(), (b, y_train.len()), &self.device)?;
        let num_present = if task == TaskType::Multiclass {
            (y_train.iter().fold(0.0f32, |a, &v| if v.is_finite() { a.max(v) } else { a }) as usize + 1).max(1)
        } else {
            0
        };
        let out = self.heads_forward(cache.decoder_keys.as_ref(), &x, &y_bn, task, num_present)?;
        let flat = out.to_vec3::<f32>()?;
        Ok(flat.into_iter().flat_map(|m| m.into_iter().flat_map(|r| r)).collect())
    }
}
