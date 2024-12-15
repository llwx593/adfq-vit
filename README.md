# ADFQ-ViT: Activation-Distribution-Friendly Post-Training Quantization for Vision Transformers 

This repository contains the code for ADFQ-ViT



## Abstract

We propose a novel framework called Activation-Distribution-Friendly post-training Quantization for Vision Transformers, ADFQ-ViT. Concretely, we introduce the Per-Patch Outlier-aware Quantizer to tackle irregular outliers in post-LayerNorm activations. This quantizer refines the granularity of the uniform quantizer to a per-patch level while retaining a minimal subset of values exceeding a threshold at full-precision. To handle the non-uniform distributions of post-GELU activations between positive and negative regions, we design the Shift-Log2 Quantizer, which shifts all elements to the positive region and then applies log2 quantization. Moreover, we present the Attention-score enhanced Module-wise Optimization which adjusts the parameters of each quantizer by reconstructing errors to further mitigate quantization error. 



## Usage

```
CUDA_VISIBLE_DEVICES=0 python -u test_quant.py --dataset /pathto/imagenet/ --model deit_small --n_bits_w 4 --n_bits_a 4 --weight 0.01 --T 4.0 --lamb_c 0.02 --iters_w 3000 --outlier 10
```

**Note 1:** Our work relies on version `timm-0.4.12`. Since our method requires access to the attention scores, a modification is needed in the `timm` library. Specifically, line 213 in `models/vision_transformer.py` should be updated as follows:

```
x = x + self.drop_path(self.attn(self.norm1(x))[0])
```

and line 293 in models/swin_transformer.py:

```
attn_windows = attn_windows[0].view(-1, self.window_size, self.window_size, C)
```

**Note 2:** The currently open-sourced code corresponds to the initial version and will be further improved in subsequent updates.

