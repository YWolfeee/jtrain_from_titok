"""This file contains the model definition of TiTok.

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
    https://github.com/facebookresearch/DiT/blob/main/models.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modules import BaseModel
from functools import partial
from timm.layers import Mlp
from typing import Optional
import numpy as np
import random

# util function
def build_causal_mask(seq_length):
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

# weight init
def init_weights(module):
    if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or
     isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                assert N in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
    

# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )


    def forward(self, x: torch.Tensor, attn_mask=None, c = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class RAR(BaseModel):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # parse the configs
        embed_dim = config.model.generator.hidden_size
        depth = config.model.generator.num_hidden_layers
        num_heads = config.model.generator.num_attention_heads
        intermediate_size = config.model.generator.intermediate_size
        mlp_ratio = intermediate_size / embed_dim

        # traning mode
        try:
            self.modelling = config.model.generator.modelling
        except:
            self.modelling = "ar"

        image_seq_len = config.model.generator.image_seq_len
        target_codebook_size = config.model.vq_model.codebook_size
        condition_num_classes = config.model.generator.condition_num_classes
        norm_layer=partial(nn.LayerNorm, eps=1e-6)

        dropout_rate = config.model.generator.dropout
        attn_dropout_rate = config.model.generator.attn_drop
   
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.embeddings = nn.Embedding(
            target_codebook_size + 1 + condition_num_classes + 1, embed_dim)

        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        self.target_aware_pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        # number of steps == image_seq_len
        self.timesteps_embeddings = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 100, embed_dim)), 0., 0.02)
        self.adaln_before_head = FinalLayer(embed_dim, norm_layer=norm_layer)
        self.lm_head = nn.Linear(embed_dim,
                                 target_codebook_size, bias=True)
        self.condition_num_classes = condition_num_classes
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1
        self.eos_id = self.condition_num_classes + self.target_codebook_size + 2
        self.pad_id = self.condition_num_classes + self.target_codebook_size + 3
        
        self.apply(init_weights)

        attn_mask = build_causal_mask(self.image_seq_len + 1024) # Guess 1024 here for safety?
        self.register_buffer('attn_mask', attn_mask, persistent=False) # enable self.attn

        self.use_checkpoint = config.model.generator.get("use_checkpoint", False)

        # init for adaln-zero.

        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        self.random_ratio = 0.0

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def sample_orders(self, x, mask_rate=None):
        batch_size = x.shape[0]
        shuffled_orders = []
        if mask_rate is None:
            mask_rate = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        for i in range(batch_size):
            kept_tokens = int((1 - mask_rate[i]) * self.image_seq_len)
            if random.random() < self.random_ratio:
                # random order for kept tokens, rest in order
                perm = torch.randperm(kept_tokens, device=x.device)
                rest = torch.arange(kept_tokens, self.image_seq_len, device=x.device)
                shuffled_orders.append(torch.cat([perm, rest]))
            else:
                # raster order
                shuffled_orders.append(torch.arange(self.image_seq_len, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)
    
    def set_random_ratio(self, new_ratio):
        self.random_ratio = new_ratio

    def get_raster_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = torch.stack([torch.arange(self.image_seq_len, device=x.device) for _ in range(batch_size)])
        return shuffled_orders

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def preprocess_condition(self, condition, cond_drop_prob=0.0):
        # condition: batch of class ids in shape of [B,], e.g. [54, 521, 69, 378];
        # To enable classifier-free guidance, create a mask that randomly drop class id;
        # E.g., if the generated mask is [0, 1, 0, 0], the second class will be replaced later.
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        # Shift the class token id to leave sapce for learned quantized tokens;
        # E.g. with 1024 ids serving for learned quantized tokens, [54, 521, 69, 378] -> [1079, 1546, 1094, 1403].
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999], here additional 1 for mask token
        # Using the generated mask to replace corresponding class id;
        # E.g. [1079, 2025, 1094, 1403], 2025 obtained by 1000 + 1024 + 1
        condition[drop_label_mask] = self.none_condition_id
        return condition
    
    def preprocess_input_ids(self, input_ids, mask_rate):
        # E.g., input_ids: [[51, 12, 64, 21], [6, 195, 87, 40]]; mask_rate: [0.0, 0.25]
        # Postpend a placeholder token for handling 0.0 mask_rate
        batch_size = input_ids.shape[0]
        # input_ids: [[51, 12, 64, 21, 2027], [6, 195, 87, 40, 2027]], 2027 -> pad_id
        input_ids = torch.cat([input_ids, torch.full((batch_size, 1), self.pad_id, device=input_ids.device)])
        kept_tokens_len = ((1 - mask_rate) * self.image_seq_len).long() # (B,)
        
        # Create position indices for comparison
        positions = torch.arange(self.image_seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, seq_len + 1]
        kept_tokens_len = kept_tokens_len.unsqueeze(1)  # [B, 1]

        # Use torch.where to replace tokens
        input_ids = torch.where(positions == kept_tokens_len, self.eos_id, input_ids)
        input_ids = torch.where(positions > kept_tokens_len, self.pad_id, input_ids)
        loss_weight_mask = torch.where(positions <= kept_tokens_len, 1, 0) # [B, S+1]

        # E.g., input_ids: [[51, 12, 64, 21, 2026], [6, 195, 87, 2026, 2027]] # 2026 -> eos_id
        return input_ids, loss_weight_mask

    def get_none_condition(self,
                           condition
                           ):
        return torch.full_like(condition, self.none_condition_id)
    
    def forward(self, input_ids, condition, mask_rate=None,return_labels=False):
        assert self.modelling in ["rar", "ar"]
        orders = self.sample_orders(input_ids, mask_rate=mask_rate) if self.modelling == "rar" else None
        return self.forward_fn(input_ids, condition, return_labels, orders)

    def forward_fn(self, input_ids, condition,
                   return_labels=False,
                   orders=None,
                   is_sampling=False,
                   adaptive_len=False,
                   mask_rate=None):
        # Token ID Correspondance:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label
        #  codebook_size + 1 + nclass + 1               : the EoS token for adaptive tokenizer
        #  codebook_size + 1 + nclass + 1 + 1           : the Padding token for adaptive tokenizer

        if orders is None:
            # We can simply convert RAR to AR by setting orders as None before forwarding
            orders = self.get_raster_orders(input_ids)

        loss_weight_mask = torch.ones_like(input_ids)
        if adaptive_len:
            # Here we preprocess input_ids to handling various length
            # E.g. token sequence [77, 49, 53, 69] w/ mask_rate=0.25 -> [77, 49, 53, 2026, 2027], 2026 - eos; 2027 - pad
            input_ids, loss_weight_mask = self.preprocess_input_ids(input_ids, mask_rate)
        
        # Input_ids are in [B,S], input_ids[i] means ith image represented by token sequence; S is the number of tokens representing image
        labels = input_ids.clone() 
        # prepend condition token, [B,S] -> [B,1+S], e.g. class id 412, with token sequence [77, 49, 53, 69] -> [412+1025, 77, 49, 53, 69];
        # If adaptive, [B+S] -> [B, 1+S+1], e.g., [412+1025, 77, 49, 53, 2026, 2027], 2026 - eos; 2027 - pad
        input_ids = torch.cat([condition.view(condition.shape[0], -1),
                            input_ids.view(input_ids.shape[0], -1),
                            ], dim=1)
        
        # mapping token ids into [B,S+1,embed_dim] or [B, S+2, embed_dim], here embed_dim is 768 in default settings
        embeddings = self.embeddings(input_ids)
        condition_token = embeddings[:, 0]

        # prepare positional embeddings.
        # shuffle pos embed
        pos_embed = self.pos_embed.repeat(input_ids.shape[0], 1, 1)
        # cls_token, condition, the permute does not impact these prefix tokens, itself prepend cls_token later
        prefix = 2
        pos_embed_prefix = pos_embed[:, :prefix]
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix:prefix+self.image_seq_len], orders)

        # prepare target-aware positional embeddings.
        target_aware_pos_embed = self.target_aware_pos_embed.repeat(input_ids.shape[0], 1, 1)
        # target_aware_pos_embed_prefix = target_aware_pos_embed[:, :prefix]
        target_aware_pos_embed_postfix = self.shuffle(target_aware_pos_embed[:, prefix:prefix+self.image_seq_len], orders)

        if not is_sampling:
            # shuffle labels
            labels = self.shuffle(labels, orders)
            # randomized permutation: during training, we need to shuffle the input_ids's order but not for sampling, but do not shuffle EoS
            if not adaptive_len:
                embeddings = torch.cat([embeddings[:, :1], self.shuffle(embeddings[:, 1:], orders)], dim=1)
            else:
                embeddings = torch.cat([embeddings[:, :1], self.shuffle(embeddings[:, 1:-1], orders), embeddings[:, -1:]], dim=1)

        x = embeddings
        # prepend the cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, 1+S, emb_dim] -> [B, 1+1+S, emb_dim] / [B, 1+S+1, emb_dim] -> [B, 1+1+S+1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)

        # add original pos embed
        x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, :x.shape[1]]

        # add target-aware pos embed
        if not adaptive_len:
            # the last permuted token not requiring target as well
            target_aware_pos_embed = torch.cat(
                [torch.zeros_like(x[:, :prefix-1]), target_aware_pos_embed_postfix, torch.zeros_like(x[:, -1:])], dim=1
            )
        else:
            # the last permuted tokens / the additional token not requiring target
            target_aware_pos_embed = torch.cat(
                [torch.zeros_like(x[:, :prefix-1]), target_aware_pos_embed_postfix, torch.zeros_like(x[:, -2:])], dim=1
            )
        x = x + target_aware_pos_embed[:, :x.shape[1]]

        # causal attention masking
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
        
        # seperate condition token for each step, at generation, we start from 1 to seq len
        condition_token = condition_token.unsqueeze(1) + self.timesteps_embeddings[:, :x.shape[1]]

        if self.blocks[0].attn.kv_cache:
            if self.blocks[0].attn.k_cache is not None and self.blocks[0].attn.v_cache is not None:
                # only need to process the last token
                x = x[:, -1:]
                attn_mask = None
                # only keep the last condition
                condition_token = condition_token[:, -1:]

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                        blk.forward, x, attn_mask, condition_token, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask, c=condition_token)

        if not self.blocks[0].attn.kv_cache:
            # remove cls token
            x = x[:, prefix - 1:] # [B, 1+1+S+1, emb_dim] -> [B, 1+S+1, embed_dim]
            condition_token = condition_token[:, prefix - 1:]


        x = self.adaln_before_head(x, condition_token)
        x = self.lm_head(x)

        # x: (B, 1+S+1, codebook_size); label: (B, S+1); loss_weight_mask: (B, S+1)
        loss_weight_mask = loss_weight_mask.to(x.dtype, x.device)
        if return_labels:
            return x, labels, loss_weight_mask
        return x
    
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale,
                 randomize_temperature,
                 guidance_scale_pow,
                 kv_cache=True,
                 **kwargs):
        condition = self.preprocess_condition(
            condition, cond_drop_prob=0.0)
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, 0), -1, device=device)
        cfg_scale = 0.

        if kv_cache:
            self.enable_kv_cache()

        orders = None
        cfg_orders = None

        for step in range(self.image_seq_len):
            # ref: https://github.com/sail-sg/MDT/blob/441d6a1d49781dbca22b708bbd9ed81e9e3bdee4/masked_diffusion/models.py#L513C13-L513C23
            scale_pow = torch.ones((1), device=device) * guidance_scale_pow
            scale_step = (1 - torch.cos(
                ((step / self.image_seq_len) ** scale_pow) * torch.pi)) * 1/2
            cfg_scale = (guidance_scale - 1) * scale_step + 1

            if guidance_scale != 0:
                logits = self.forward_fn(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0),
                    orders=cfg_orders, is_sampling=True)
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward_fn(
                    ids, condition, orders=orders, is_sampling=True
                )

            # keep the logit of last token
            logits = logits[:, -1]
            logits = logits / randomize_temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, sampled), dim = -1)


        self.disable_kv_cache()
        return ids
    