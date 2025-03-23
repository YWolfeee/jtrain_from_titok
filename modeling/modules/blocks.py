"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import einops
import math
from einops.layers.torch import Rearrange


def modulate(x, shift, scale):
    """used for AdaLN-Zero.
    """
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class ResidualAttentionBlock(nn.Module):
    """Use AdaLN-Zero to make attention block conditioned on timestep embedding.
    """
    def __init__(
        self,
        d_model,
        n_head,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_temb=False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, mlp_width)),
                        ("gelu", act_layer()),
                        ("c_proj", nn.Linear(mlp_width, d_model)),
                    ]
                )
            )
        self.use_temb = use_temb
        if use_temb:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True)
            )

    def attention(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ):
        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )[0]

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
    ):
        if self.use_temb:
            assert temb is not None, "temb must be provided if use_temb is True"
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(temb).chunk(6, dim=1)
            )
            # broadcast across sequence, x.shape: (seq_len, batch_size, d_model)
            x = x + gate_msa.unsqueeze(0) * self.attention(
                x=modulate(self.ln_1(x), shift_msa, scale_msa),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            x = x + gate_mlp.unsqueeze(0) * self.mlp(
                modulate(self.ln_2(x), shift_mlp, scale_mlp)
            )
        else:
            attn_output = self.attention(
                x=self.ln_1(x), key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            x = x + attn_output
            if self.mlp_ratio > 0:
                x = x + self.mlp(self.ln_2(x))
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=t.dtype)
            / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"
print(f"attention mode is {ATTENTION_MODE}")


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == "flash":
            qkv = einops.rearrange(
                qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
            ).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "xformers":
            qkv = einops.rearrange(
                qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.num_heads)
        elif ATTENTION_MODE == "math":
            qkv = einops.rearrange(
                qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class REPAProjection(nn.Module):
    """This is reproduced from REPA: https://arxiv.org/abs/2410.06940
    
    The REPAProjection accepts a feature in the FlowDecoder and projects it to the space of dino feature.
    """
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class UViTBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.grid_size = (
            self.image_size // config.model.vq_model.vit_enc_patch_size
        )  # 16
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.from_continuous = config.model.vq_model.get("from_continuous", False)

        if self.from_continuous:
            self.in_channels = 16  # VAE latent channel
            self.vae_downsample_factor = (
                8  # We use VAE encoding image from [1, 3, 256, 256] -> [1, 16, 32, 32]
            )
            self.patch_size = (
                config.model.vq_model.vit_enc_patch_size // self.vae_downsample_factor
            )
        else:
            self.in_channels = 3  # RGB channel
            self.patch_size = config.model.vq_model.vit_enc_patch_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2  # needs to split into mean and std

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
            "huge": 1280,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
            "huge": 32,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
            "huge": 20,
        }[self.model_size]

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size**2 + 1, self.width)
        )
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            )
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(
        self, pixel_values, latent_tokens, key_padding_mask=None, attn_mask=None
    ):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat(
            [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1
        )
        x = x + self.positional_embedding.to(
            x.dtype
        )  # shape = [*, grid ** 2 + 1, width]

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(
            x.dtype
        )
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](
                x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        x = x.permute(1, 0, 2)  # LND -> NLD

        latent_tokens = x[:, 1 + self.grid_size**2 :]
        latent_tokens = self.ln_post(latent_tokens)
        latent_embeddings = latent_tokens.clone()
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(
                batch_size, self.width, self.num_latent_tokens, 1
            )
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(
                batch_size, self.num_latent_tokens, self.width, 1
            ).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(
            batch_size, self.token_size, 1, self.num_latent_tokens
        )
        return latent_tokens, latent_embeddings


class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.from_continuous = config.model.vq_model.get("from_continuous", False)
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
            "huge": 1280,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
            "huge": 32,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
            "huge": 20,
        }[self.model_size]

        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size**2 + 1, self.width)
        )
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            )
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy and not self.from_continuous:
            # Transform to the shape of [B, 1024, 16, 16] for MasGiT-VQGAN
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        elif self.is_legacy and self.from_continuous:
            # Transform to the shape of [B, 16, 32, 32] for FLUX VAE
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 64, 1, padding=0, bias=True),
                Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=2, p2=2),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(
                    self.width,
                    self.patch_size * self.patch_size * 3,
                    1,
                    padding=0,
                    bias=True,
                ),
                Rearrange(
                    "b (p1 p2 c) h w -> b c (h p1) (w p2)",
                    p1=self.patch_size,
                    p2=self.patch_size,
                ),
            )
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

    def forward(self, z_quantized, key_padding_mask=None, attn_mask=None):
        N, C, H, W = z_quantized.shape
        assert (
            H == 1 and W == self.num_latent_tokens
        ), f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C * H, W).permute(0, 2, 1)  # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(
            x.dtype
        )
        mask_tokens = torch.cat(
            [
                _expand_token(self.class_embedding, mask_tokens.shape[0]).to(
                    mask_tokens.dtype
                ),
                mask_tokens,
            ],
            dim=1,
        )
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](
                x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1 : 1 + self.grid_size**2]  # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(
            batchsize, self.width, self.grid_size, self.grid_size
        )
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x


class FlowDecoder(nn.Module):
    """This is reproduced from FlexTok: https://www.arxiv.org/pdf/2502.13967

    The FlowDecoder accepts noisy vae latent, a timestep information and a key padding mask.
    It outputs the predicted flow and the repa feature to be aligned with dino feature.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.from_continuous = config.model.vq_model.get("from_continuous", False)
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
            "huge": 1280,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
            "huge": 32,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
            "huge": 20,
        }[self.model_size]
        self.repa_depth = 1
        self.repa_projection = REPAProjection(
            self.width, self.width, 1024
        )  # 1024 for large

        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size**2 + 1, self.width)
        )
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            # Include time embedding for denoising
            self.transformer.append(
                ResidualAttentionBlock(
                    self.width, self.num_heads, mlp_ratio=4.0, use_temb=True
                )
            )
        self.ln_post = nn.LayerNorm(self.width)

        # Transform to the shape of [B, 16, 32, 32] for FLUX VAE
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(2 * self.width, 64, 1, padding=0, bias=True),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=2, p2=2),
        )
        self.conv_out = nn.Identity()

        # Timestep Embedding
        self.temb_layer = TimestepEmbedder(self.width)

        # Patchify vae latent: (B, 16, 32, 32) -> (B, self.width, 16, 16)
        self.patch_embed = nn.Conv2d(
            in_channels=16, out_channels=self.width, kernel_size=2, stride=2, bias=True
        )

    def forward(
        self, z_quantized, noisy_vae_latent, t, key_padding_mask=None, attn_mask=None
    ):
        N, C, H, W = z_quantized.shape
        assert (
            H == 1 and W == self.num_latent_tokens
        ), f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C * H, W).permute(0, 2, 1)  # NLD
        x = self.decoder_embed(x)

        temb = self.temb_layer(t)
        batchsize, seq_len, _ = x.shape

        noisy_vae_latent = self.patch_embed(noisy_vae_latent)
        noisy_vae_latent = noisy_vae_latent.reshape(N, self.width, -1).permute(
            0, 2, 1
        )  # NLD
        noisy_vae_latent = torch.cat(
            [
                _expand_token(self.class_embedding, N).to(noisy_vae_latent.dtype),
                noisy_vae_latent,
            ],
            dim=1,
        )
        noisy_vae_latent = noisy_vae_latent + self.positional_embedding.to(
            noisy_vae_latent.dtype
        )
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([noisy_vae_latent, x], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](
                x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, temb=temb
            )
            if i + 1 == self.repa_depth:
                repa_feature = self.repa_projection(
                    x[: 1 + self.grid_size**2, :]
                ).permute(
                    1, 0, 2
                )  # NLD
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1 : 1 + self.grid_size**2]  # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(
            batchsize, self.width, self.grid_size, self.grid_size
        )
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x, repa_feature


class PolicyNet(nn.Module):
    """Policy indicating the token length used for tokenizer.

    If self.elbo.nll_only, it will be fully ruled-based policy without parameterized neural network. 
    """
    def __init__(
        self,
        config,
        in_channels,
        num_tokens,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.hidden_size = config.model.reconstruction_regularization.policy.hidden_size

        try:
            self.elbo = config.model.reconstruction_regularization.policy.elbo
        except:
            self.elbo = None

        if self.elbo.nll_only:
            # In this case, no need for any neural network
            return

        self.model_type = config.model.reconstruction_regularization.policy.model_type
        assert self.model_type in [
            "mlp",
            "transformer",
            "causal_transformer",
        ], "model_type must be either mlp / transformer / causal_transformer"

        if self.model_type == "mlp":
            self.fc1 = nn.Linear(self.in_channels, self.hidden_size)

        elif (
            self.model_type == "transformer" or self.model_type == "causal_transformer"
        ):
            self.num_heads = config.model.reconstruction_regularization.policy.num_heads
            self.num_layers = num_layers
            self.positional_embedding = nn.Parameter(
                torch.randn(1, self.num_tokens, self.in_channels)
            )
            self.ln_pre = nn.LayerNorm(self.in_channels)
            self.transformer = nn.ModuleList()
            for i in range(self.num_layers):
                self.transformer.append(
                    ResidualAttentionBlock(
                        self.in_channels, self.num_heads, mlp_ratio=mlp_ratio
                    )
                )
            self.ln_post = nn.LayerNorm(self.in_channels)

        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Logit prediction
        self.logit_head_type = (
            config.model.reconstruction_regularization.policy.logit_head_type
        )
        assert self.logit_head_type in [
            "categorical_256",
            "categorical_8",
            "gaussian_1",
        ], "logit_head must be either categorical_256 / categorical_8 / gaussian_1"
        last_hidden_size = (
            self.hidden_size if self.model_type == "mlp" else self.in_channels
        )
        if self.logit_head_type == "categorical_256":
            self.logit_head = nn.Linear(last_hidden_size, 256)
        elif self.logit_head_type == "categorical_8":
            self.logit_head = nn.Linear(last_hidden_size, 8)
        elif self.logit_head_type == "gaussian_1":
            self.logit_head = nn.Linear(last_hidden_size, 1)

    def forward(
        self,
        dino_feature: torch.Tensor,
        temperature=1.0,
        annealing_factor=1.0,
        gaussian_sampling_sigma=1.0,
        vae_results: dict = None,  # For pre-get NLL to constrain the mask rate
    ):
        if self.elbo and self.elbo.nll_only:
            if vae_results is None:
                elbo = torch.ones((dino_feature.shape[0],)).to(dino_feature.device)
            else:
                elbo = vae_results["elbo"] / vae_results["elbo_avg"]

            mask_rate = 1 - self.elbo.mean * elbo
            mode = self.elbo.elbo_mode  # remove implicit mode

            assert self.elbo.mean == 0.5, "current code does not implement logis beyond mean==0.5."
            if mode == "px":
                mask_rate = mask_rate
            elif mode == "titok":
                mask_rate = 1 - self.elbo.mean * torch.ones_like(mask_rate)
            elif mode == "elastic":
                mini_val = 1 / 16
                mask_rate = (1 - mini_val) * (torch.rand_like(mask_rate))
            elif mode == "flextok":
                # generate all possible 2**i not larger than self.num_tokens
                total_length = int(math.log2(self.num_tokens)) + 1
                candidates = torch.tensor(
                    [2 ** i/self.num_tokens for i in range(total_length)]
                ).to(mask_rate.device)
                mask_rate = 1 - candidates[torch.randint_like(
                    mask_rate, 0, total_length).int()]
            else:
                raise NotImplementedError("Unrecognized elbo_mode value.")

            # anneal with annealing_factor, globally
            mask_rate = annealing_factor * mask_rate + (1 - annealing_factor) * (
                1 - self.elbo.start_mean
            )

            mask_rate = mask_rate.clip(
                self.elbo.get("lower", 0.0), self.elbo.get("upper", 1.0)
            )

            return {
                "sampled_mask_rate": mask_rate,
                "mask_rate_value": mask_rate,
                "logprob_mask": elbo,
            }

        if self.model_type == "mlp":
            global_token = dino_feature[:, 0, :]  # [B, C]
            x = nn.functional.gelu(
                self.fc1(global_token)
            )  # Use the first global token [cls_token]
            logits = self.logit_head(x)

        elif self.model_type == "transformer":
            dino_feature = dino_feature + self.positional_embedding
            dino_feature = self.ln_pre(dino_feature)
            dino_feature = dino_feature.permute(1, 0, 2)
            for i in range(self.num_layers):
                dino_feature = self.transformer[i](dino_feature)
            dino_feature = dino_feature.permute(1, 0, 2)
            dino_feature = self.ln_post(dino_feature)
            global_token = dino_feature[:, 0, :]  # [B, C]
            logits = self.logit_head(global_token)

        elif self.model_type == "causal_transformer":
            N = dino_feature.shape[1]
            dino_feature = dino_feature + self.positional_embedding
            dino_feature = self.ln_pre(dino_feature)
            causal_mask = (
                torch.triu(torch.ones(N, N), diagonal=1).bool().to(dino_feature.device)
            )
            dino_feature = dino_feature.permute(1, 0, 2)
            for i in range(self.num_layers):
                dino_feature = self.transformer[i](dino_feature, src_mask=causal_mask)
            dino_feature = dino_feature.permute(1, 0, 2)
            dino_feature = self.ln_post(dino_feature)
            global_token = dino_feature[:, 0, :]  # [B, C]
            logits = self.logit_head(global_token)

        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        if (
            self.logit_head_type == "categorical_256"
            or self.logit_head_type == "categorical_8"
        ):  # Categorical sampling
            logits = logits - torch.mean(logits, dim=-1, keepdim=True)

            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  # [B, N]
            samples = torch.multinomial(probs, num_samples=1)[:, 0]
            sampled_prob = probs[torch.arange(samples.shape[0]), samples]
            mask_rate = (
                1 - (samples + 1) / probs.shape[1]
            )  # Resolve 8 categories and 256 categories

            return {
                "sampled_mask_rate": mask_rate,
                "mask_rate_value": mask_rate,
                "logprob_mask": torch.log(sampled_prob),
            }

        elif self.logit_head_type == "gaussian_1":  # Gaussian sampling
            # Use truncated normal distribution
            rate_mean = torch.sigmoid(logits)[:, 0]

            normal = torch.distributions.Normal(rate_mean, gaussian_sampling_sigma)

            # Convert bounds to tensors on the same device & dtype as mean.
            a_tensor = torch.tensor(0, dtype=rate_mean.dtype, device=rate_mean.device)
            b_tensor = torch.tensor(1, dtype=rate_mean.dtype, device=rate_mean.device)

            # Compute the CDF values at the truncation bounds.
            cdf_a = normal.cdf(a_tensor)
            cdf_b = normal.cdf(b_tensor)

            # Sample uniform values in [cdf_a, cdf_b] for each batch element.
            u = torch.empty_like(rate_mean).uniform_(0, 1)
            u_scaled = u * (cdf_b - cdf_a) + cdf_a  # maps to [cdf(a), cdf(b)]

            # Use the inverse CDF (icdf) to obtain the sample. We do not want to backprop through this.
            sample_rate = normal.icdf(u_scaled).detach()
            sampled_mask_rate = 1 - sample_rate  # ratio of masking

            # This is the un-normalized log-prob that use for pairwise reinforce
            logprob_mask = -((sample_rate - rate_mean) ** 2) / (
                2 * gaussian_sampling_sigma**2
            )

            return {
                "sampled_mask_rate": sampled_mask_rate,
                "mask_rate_value": sampled_mask_rate,
                "logprob_mask": logprob_mask,
            }
        else:
            raise ValueError(f"Invalid logit head type: {self.logit_head_type}")
