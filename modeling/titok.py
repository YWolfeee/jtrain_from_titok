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
"""

import torch
import torch.nn as nn
from einops import rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
            
        # QY: Add regularization for using partial tokens for reconstruction
        if config.model.use_reconstruction_regularization:
            self.use_regularization = True
            self.regularization_name = config.model.reconstruction_regularization.name
            self.mask_ratio_method = config.model.reconstruction_regularization.mask_ratio_method
            self.max_mask_rate = config.model.reconstruction_regularization.max_mask_rate
        else:
            self.use_regularization = False
            self.max_mask_rate = 0.0
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_max_mask_rate(self, max_mask_rate):
        self.max_mask_rate = max_mask_rate

    def encode(self, x, drop_p=0.0):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict

    def get_mask_rate(self, z_quantized, decode_mask_rate=0.0):
        device = z_quantized.device
        if self.use_regularization and self.training:
            if self.mask_ratio_method == "uniform":
                mask_rate = torch.empty(z_quantized.shape[0], device=device).uniform_(0, self.max_mask_rate - 1e-3)
            elif self.mask_ratio_method == "hierarchical":
                values = torch.tensor([self.max_mask_rate * i / 16 for i in range(16)], device=device)  # we do not consider zero-token setting
                indices = torch.randint(0, values.shape[0], (z_quantized.shape[0],), device=device)
                mask_rate = values[indices]
            else:
                raise NotImplementedError(f"Unsupported mask ratio method {self.mask_ratio_method}.")
        else:
            mask_rate = torch.tensor(decode_mask_rate, device=device).expand(z_quantized.shape[0])

        return mask_rate
    
    def decode(self, z_quantized, decode_mask_rate=None):
        if isinstance(decode_mask_rate, float):
            decode_mask_rate = torch.tensor(decode_mask_rate, device=z_quantized.device).expand(z_quantized.shape[0])

        # mask rate is a tensor with shape (z_quantized.shape[0],)
        # values could be identical inside            
        if self.regularization_name == "matryoshka":
            z_quantized = self.matryoshka_masking(z_quantized, mask_rate=decode_mask_rate)
        elif self.regularization_name == "random":
            raise NotImplementedError(
                "This training approach has been deprecated.")
            z_quantized = self.random_masking(z_quantized, mask_rate=decode_mask_rate)
        else:
            raise NotImplementedError(f"Unsupported reconstruction regularization {self.reconstruction_regularization}.")
        
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)

        return decoded
    
    def decode_tokens(self, tokens, decode_mask_rate=0.0):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decode_mask_rate = self.get_mask_rate(z_quantized, decode_mask_rate)
        decoded = self.decode(z_quantized, decode_mask_rate=decode_mask_rate)
        return decoded
    
    def matryoshka_masking(self, z_quantized, mask_rate):
        # outside function should ensure that mask_rate is meaningful
        # e.g. belong to [0, 1)
        keep_tokens = torch.ceil(z_quantized.shape[-1] * (1 - mask_rate)).long().to(z_quantized.device)

        mask = torch.arange(z_quantized.shape[-1], device=z_quantized.device)[None] < keep_tokens[:, None]
        return torch.where(mask[:, None, None], z_quantized, 0)
    
    def random_masking(self, z_quantized, mask_rate):
        mask = torch.rand_like(z_quantized[0:1, 0:1, 0:1, :]) > mask_rate
        z_quantized = z_quantized * mask.to(z_quantized.dtype, z_quantized.device)
        return z_quantized
    
    def forward(self, x, decode_mask_rate=0.0):
        if not isinstance(decode_mask_rate, float):
            raise ValueError("decode_mask_rate in forward() should be a float")
        
        z_quantized, result_dict = self.encode(x)
        decode_mask_rate = self.get_mask_rate(z_quantized, decode_mask_rate)
        decoded = self.decode(z_quantized, decode_mask_rate=decode_mask_rate)
        if self.config.losses.use_self_distilliation:
            with torch.no_grad():
                result_dict["decode_mask_rate"] = decode_mask_rate
                result_dict["self_distilliated_codes"] = self.decode(z_quantized, max(0, decode_mask_rate - 1/16))
        return decoded, result_dict
