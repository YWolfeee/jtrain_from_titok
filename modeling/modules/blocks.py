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
from einops.layers.torch import Rearrange
from einops import rearrange

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

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
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        latent_embeddings = latent_tokens.clone()
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens, latent_embeddings
    

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, config, in_channels: int, num_layers: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.in_channels = in_channels
        self.hidden_size = config.model.reconstruction_regularization.policy.hidden_size

        self.model_type = config.model.reconstruction_regularization.policy.model_type
        assert self.model_type in ["mlp", "transformer", "causal_transformer"], "model_type must be either mlp / transformer / causal_transformer"
        
        if self.model_type == "mlp":
            self.fc1 = nn.Linear(self.in_channels, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, 1)

        elif self.model_type == "transformer" or self.model_type == "causal_transformer":
            self.num_heads = config.model.reconstruction_regularization.policy.num_heads
            self.num_layers = num_layers
            self.positional_embedding = nn.Parameter(torch.randn(1, self.num_latent_tokens, self.in_channels))
        
            # Single-layer transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.in_channels,
                nhead=self.num_heads,
                dim_feedforward=int(self.in_channels * mlp_ratio),
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            
            # Logit prediction
            self.logit_head = nn.Linear(self.in_channels, 1)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def forward(self, 
                z_embeddings: torch.Tensor, 
                temperature=1.0, 
                gumbel_softmax=None, 
                gaussian_smoothing=None, 
                annealing_factor=1.0
        ):
        try:
            self.use_pairwise = self.config.model.reconstruction_regularization.policy.use_pairwise
        except:
            self.use_pairwise = False
        # DEBUG: print(f"\033[91mCHECK temperature", temperature, "\033[0m")
        if self.model_type == "mlp":
            # batch_size, self.in_channels, self.num_latent_tokens
            B, _, _ = z_embeddings.shape
            # DEBUG: print("\033[91mCHECK the shape of z_quantized", z_quantized.shape, "\033[0m")
            # z_flattened = rearrange(z_quantized, 'b c w -> b w c').contiguous()
            z_flattened = rearrange(z_embeddings, 'b w c -> (b w) c') # reshape as (b*h*w, c)
            x = nn.functional.silu(self.fc1(z_flattened))
            x = self.fc2(x) # (b*w, 1)
            # reshape back to (b, w)
            logits = x.reshape(B, -1)
            # DEBUG: print("\033[91mCHECK the shape of logits", logits.shape, "\033[0m")
            # apply softmax
        
        elif self.model_type == "transformer":
            B, N, D = z_embeddings.shape
            # z_quantized = z_quantized.squeeze(2).transpose(1, 2)  # [B, N, D]
            
            # Add positional embeddings
            z_embeddings = z_embeddings + self.positional_embedding
            
            # Apply transformer
            # z_embeddings = z_embeddings.permute(1, 0, 2) # [N, B, D]
            features = self.transformer(z_embeddings)  # [B, N, D]
            # features = features.permute(1, 0, 2) # [B, N, D]
            # Predict logits
            logits = self.logit_head(features).squeeze(-1)  # [B, N]
        
        elif self.model_type == "causal_transformer":
            B, N, D = z_embeddings.shape
            # z_embeddings = z_embeddings.squeeze(2).transpose(1, 2)  # [B, N, D]
            
            # Add positional embeddings
            z_embeddings = z_embeddings + self.positional_embedding
            
            # Apply causal transformer
            causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool().to(z_embeddings.device)
            # z_embeddings = z_embeddings.permute(1, 0, 2) # [N, B, D]
            features = self.transformer(z_embeddings, src_mask=causal_mask)  # [N, B, D]
            # features = features.permute(1, 0, 2) # [B, N, D]
            
            # Predict logits
            logits = self.logit_head(features).squeeze(-1)  # [B, N]
        
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        # Gaussian smoothing
        if gaussian_smoothing is not None:
            logits = apply_gaussian_smoothing(logits, gaussian_smoothing.kernel_size, gaussian_smoothing.sigma)

        # Normalize logits
        try:
            normalize_logits = self.config.model.reconstruction_regularization.policy.normalize_logits
        except:
            normalize_logits = False
        if normalize_logits:
            logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        
        if gumbel_softmax is not None: # we don't do reinforce
            # This is actually a vector of shape (btz, max_code_length)
            logits = annealing_factor * logits + \
                (1-annealing_factor) * logits.detach()
            if gumbel_softmax.fix_tau:
                logits /= temperature
                temperature = 1.0

            sampled_rate = torch.nn.functional.gumbel_softmax(
                    logits,
                    hard=gumbel_softmax.hard,
                    tau=temperature,
                    dim=-1)
            N = sampled_rate.shape[-1]
            MASK = torch.tril(torch.ones((N, N), device=sampled_rate.device), 
                              diagonal=0)
            sampled_rate = sampled_rate @ MASK
            return {
                "sampled_mask_rate": sampled_rate,
                "mask_rate_value": 1 - sampled_rate.mean(dim=-1)
            }
        else:
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            if not self.use_pairwise or not self.training:
                sampled_num = torch.multinomial(probs, num_samples=1)[:, 0]
                sampled_prob = probs[torch.arange(sampled_num.shape[0]),
                                        sampled_num]
            else:
                assert probs.shape[0] % 2 == 0, "batch size must be even for pairwise sampling"
                sampled_num = torch.multinomial(probs[:probs.shape[0]//2], num_samples=2) # shape: (B/2, 2)
                sampled_prob_1 = probs[torch.arange(sampled_num.shape[0]),
                                        sampled_num[:, 0]]
                sampled_prob_2 = probs[torch.arange(sampled_num.shape[0]),
                                        sampled_num[:, 1]]
                # stack sampled_num and sampled_prob
                sampled_num = torch.cat([sampled_num[:, 0], sampled_num[:, 1]], dim=0)
                sampled_prob = torch.cat([sampled_prob_1, sampled_prob_2], dim=0)
                # DEBUG: print("/033[91mCHECK sampled_num.shape", sampled_num.shape, "\033[0m")
                # DEBUG: print("/033[91mCHECK sampled_prob.shape", sampled_prob.shape, "\033[0m")
            mask_rate = 1 - sampled_num / self.num_latent_tokens

            return {
                "sampled_mask_rate": mask_rate,
                "mask_rate_value": mask_rate,
                "prob_of_sampled_mask_rate": sampled_prob
            }
        
def apply_gaussian_smoothing(logits, kernel_size=64, sigma=5.0):
    """
    Apply Gaussian smoothing to a 1D tensor of logits.

    Parameters:
    logits (torch.Tensor): The input tensor with shape [batch_size, length].
    kernel_size (int): The size of the Gaussian kernel.
    sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
    torch.Tensor: The smoothed logits tensor with the same shape as input.
    """
    # Ensure kernel_size is odd to have a symmetric kernel
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")

    # Create a 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=logits.dtype, device=logits.device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    logits = logits.unsqueeze(1)
    smoothed_logits = nn.functional.conv1d(logits, kernel, padding=kernel_size // 2)
    smoothed_logits = smoothed_logits.squeeze(1)

    return smoothed_logits
        
        