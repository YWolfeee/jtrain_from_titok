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
            x: torch.Tensor,
            key_padding_mask: torch.Tensor = None
    ):
        return self.attn(x, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: torch.Tensor = None
    ):
        attn_output = self.attention(x=self.ln_1(x), key_padding_mask=key_padding_mask)
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

    def forward(self, pixel_values, latent_tokens, key_padding_mask=None):
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
            x = self.transformer[i](x, key_padding_mask=key_padding_mask)
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
    
    def forward(self, z_quantized, key_padding_mask=None):
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
            x = self.transformer[i](x, key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, config, in_channels, num_tokens, num_layers: int = 4, mlp_ratio: float = 4.0):
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
        assert self.model_type in ["mlp", "transformer", "causal_transformer"], \
            "model_type must be either mlp / transformer / causal_transformer"
        
        if self.model_type == "mlp":
            self.fc1 = nn.Linear(self.in_channels, self.hidden_size)

        elif self.model_type == "transformer" or self.model_type == "causal_transformer":
            self.num_heads = config.model.reconstruction_regularization.policy.num_heads
            self.num_layers = num_layers
            self.positional_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.in_channels))
            # encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=self.in_channels,
            #     nhead=self.num_heads,
            #     dim_feedforward=int(self.in_channels * mlp_ratio),
            #     activation="gelu",
            #     batch_first=True,
            # )
            # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.ln_pre = nn.LayerNorm(self.in_channels)
            self.transformer = nn.ModuleList()
            for i in range(self.num_layers):
                self.transformer.append(ResidualAttentionBlock(
                    self.in_channels, self.num_heads, mlp_ratio=4.0
                ))
            self.ln_post = nn.LayerNorm(self.in_channels)
        
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Logit prediction
        self.logit_head_type = config.model.reconstruction_regularization.policy.logit_head_type
        assert self.logit_head_type in ["categorical_256", "categorical_8", "gaussian_1"], \
            "logit_head must be either categorical_256 / categorical_8 / gaussian_1"
        last_hidden_size = self.hidden_size if self.model_type == "mlp" else self.in_channels
        if self.logit_head_type == "categorical_256":
            self.logit_head = nn.Linear(last_hidden_size, 256)
        elif self.logit_head_type == "categorical_8":
            self.logit_head = nn.Linear(last_hidden_size, 8)
        elif self.logit_head_type == "gaussian_1":
            self.logit_head = nn.Linear(last_hidden_size, 1)

    def forward(self, 
                token_features: torch.Tensor, 
                temperature=1.0, 
                gumbel_softmax=None, 
                gaussian_smoothing=None, 
                annealing_factor=1.0,
                gaussian_sampling_sigma=1.0,
                use_pairwise: bool=False,
                vae_results: dict = None, # For pre-get NLL to constrain the mask rate
        ):
        # DEBUG: print parameter norm of logit_head
        # print("\033[91mCHECK parameter norm of logit_head", self.logit_head.weight.norm(), "\033[0m")
        if self.elbo and self.elbo.nll_only:   
            if vae_results is None:
                elbo =  torch.ones((token_features.shape[0],)).to(
                    token_features.device)
            else:
                elbo = vae_results['elbo'] / vae_results['elbo_avg']

            mask_rate = 1 - self.elbo.mean * elbo
            mode = self.elbo.get("elbo_mode", "")
            if mode == "0.4+0.6":
                mask_rate = torch.where(mask_rate < 0.5, 0.4, 0.6)
            elif mode == "upto_px":
                r = torch.rand_like(mask_rate)
                mask_rate += (1 - mask_rate) * r
            elif mode == "downto_px":
                mask_rate *= torch.rand_like(mask_rate)
            elif mode == "0.1_in_px":
                r = (torch.rand_like(mask_rate) - 0.5) / 0.5 * 0.1
                mask_rate += r
            elif mode == "0.1_in_0.5":
                mask_rate = (torch.rand_like(mask_rate) - 0.5) / 5 + 0.5
            elif mode == "anneal_to_px":
                start = 1 - self.elbo.start_mean
                end = mask_rate
                mask_rate = annealing_factor * end + (1-annealing_factor) * start
                
            mask_rate = mask_rate.clip(self.elbo.get('lower', 0.0), 
                                       self.elbo.get('upper', 1.0))
            
            return {
                "sampled_mask_rate": mask_rate,
                "mask_rate_value": mask_rate,
                "logprob_mask": elbo,
            }

        if self.model_type == "mlp":
            global_token = token_features[:, 0, :] # [B, C]
            x = nn.functional.gelu(self.fc1(global_token)) # Use the first global token [cls_token]
            logits = self.logit_head(x)

        elif self.model_type == "transformer":
            # DEBUG: print("\033[91mCHECK token_features.shape", token_features.shape, "\033[0m")
            # DEBUG: print("\033[91mCHECK self.positional_embedding.shape", self.positional_embedding.shape, "\033[0m")
            token_features = token_features + self.positional_embedding
            token_features = self.ln_pre(token_features)
            token_features = token_features.permute(1, 0, 2)
            for i in range(self.num_layers):
                token_features = self.transformer[i](token_features)
            token_features = token_features.permute(1, 0, 2)
            token_features = self.ln_post(token_features)
            global_token = token_features[:, 0, :] # [B, C]
            logits = self.logit_head(global_token)
        
        elif self.model_type == "causal_transformer":
            N = token_features.shape[1]
            token_features = token_features + self.positional_embedding
            token_features = self.ln_pre(token_features)
            causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool().to(token_features.device)
            token_features = token_features.permute(1, 0, 2)
            for i in range(self.num_layers):
                token_features = self.transformer[i](token_features, src_mask=causal_mask)
            token_features = token_features.permute(1, 0, 2)
            token_features = self.ln_post(token_features)
            global_token = token_features[:, 0, :] # [B, C]
            logits = self.logit_head(global_token)

        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        # Gaussian smoothing
        if gaussian_smoothing is not None:
            # assert self.logit_head_type == "categorical_256", "Gaussian smoothing is only supported for categorical_256"
            logits = apply_gaussian_smoothing(logits, gaussian_smoothing.kernel_size, gaussian_smoothing.sigma)
        
        if use_pairwise:
            # before sampling, we double the logits
            # in this case subsequent logits are consistent
            logits = torch.concat([logits, logits])

        # Use [Gumbel Softmax] or [Sampling w/ REINFORCE]
        if gumbel_softmax is not None: # We don't do reinforce
            # assert (
            #     self.logit_head_type == "categorical_256" or 
            #     self.logit_head_type == "categorical_8"
            # ), "Gumbel softmax is only supported for categorical_256 and categorical_8"
            raise NotImplementedError("Gumbel softmax has been deprecated.")
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

        elif self.logit_head_type == "categorical_256" or self.logit_head_type == "categorical_8": # Categorical sampling
            logits = logits - torch.mean(logits, dim=-1, keepdim=True)

            probs = torch.nn.functional.softmax(logits / temperature, dim=-1) # [B, N]
            samples = torch.multinomial(probs, num_samples=1)[:, 0]
            sampled_prob = probs[torch.arange(samples.shape[0]), samples]
            # else:
            #     assert probs.shape[0] % 2 == 0, "batch size must be even for pairwise sampling"
            #     sampled_num = torch.multinomial(probs[:probs.shape[0]//2], num_samples=2) # shape: (B/2, 2)
            #     sampled_prob_1 = probs[torch.arange(sampled_num.shape[0]),
            #                             sampled_num[:, 0]]
            #     sampled_prob_2 = probs[torch.arange(sampled_num.shape[0]),
            #                             sampled_num[:, 1]]
            #     # stack sampled_num and sampled_prob
            #     sampled_num = torch.cat([sampled_num[:, 0], sampled_num[:, 1]], dim=0)
            #     sampled_prob = torch.cat([sampled_prob_1, sampled_prob_2], dim=0)
            #     # DEBUG: print("/033[91mCHECK sampled_num.shape", sampled_num.shape, "\033[0m")
                # DEBUG: print("/033[91mCHECK sampled_prob.shape", sampled_prob.shape, "\033[0m")
            mask_rate = 1 - (samples + 1) / probs.shape[1] # Resolve 8 categories and 256 categories

            return {
                "sampled_mask_rate": mask_rate,
                "mask_rate_value": mask_rate,
                "logprob_mask": torch.log(sampled_prob),
            }

        elif self.logit_head_type == "gaussian_1": # Gaussian sampling

            # ### Sample then Normalize
            # print("\033[91mAlert: Deprecated implementation", "\033[0m")
            # # Reparameterize and sample from Gaussian distribution with std=temperature
            # sampled_from_logits = torch.randn_like(logits) * gaussian_sampling_sigma + logits
            # # Compute the probability of the sampled mask rate based on Gaussian distribution
            # logprob_mask = torch.exp(
            #     -0.5 * ((sampled_from_logits - logits) / gaussian_sampling_sigma) ** 2
            # ) / (gaussian_sampling_sigma * math.sqrt(2 * math.pi))
            # sampled_rate = torch.sigmoid(sampled_from_logits)[:, 0]

            # ### Use truncated normal distribution
            rate_mean = torch.sigmoid(logits)[:, 0]

            # sample_rate = torch.zeros_like(rate_mean)
            # torch.nn.init.trunc_normal_(sample_rate, rate_mean, gaussian_sampling_sigma, 0, 1)

            normal = torch.distributions.Normal(rate_mean, gaussian_sampling_sigma)
            # print("\033[91mCHECK rate_mean", rate_mean, "\033[0m")
            # print("\033[91mCHECK gaussian_sampling_sigma", gaussian_sampling_sigma, "\033[0m")
    
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
            logprob_mask_check = normal.log_prob(sample_rate)
            # print("\033[91mCHECK sampled_rate", sampled_rate, "\033[0m")
            # print("\033[91mCHECK logprob_mask", logprob_mask, "\033[0m")

            sampled_mask_rate = 1 - sample_rate    # ratio of masking

            # This is the un-normalized log-prob that use for pairwise reinforce
            logprob_mask = - (sample_rate - rate_mean)**2 / (2 * gaussian_sampling_sigma ** 2)
            
            return {
                "sampled_mask_rate": sampled_mask_rate,
                "mask_rate_value": sampled_mask_rate,
                "logprob_mask": logprob_mask,
            }
        else:
            raise ValueError(f"Invalid logit head type: {self.logit_head_type}")

        
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
        
        