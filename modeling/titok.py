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
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder, PolicyNet
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
import math
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel


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
        # Whether to freeze encoder / decoder during the training
        self.freeze_encoder = config.model.vq_model.get("freeze_encoder", False)
        self.freeze_decoder = config.model.vq_model.get("freeze_decoder", False)

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

        self.feature_extractor_name = config.model.reconstruction_regularization.policy.feature_extractor_name # 'facebook/dinov2-base'
        self.feature_extractor = AutoModel.from_pretrained(self.feature_extractor_name)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False) # OUTPUT SHAPE: [B, 257, 768] for base, [B, 257, 1024] for large

        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            if self.use_policy:
                self.policy_net.eval()
                self.policy_net.requires_grad_(False)

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
        
        if self.freeze_decoder:
            self.decoder.eval()
            self.decoder.requires_grad_(False)
        if self.freeze_encoder:
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            
        # QY: Add regularization for using partial tokens for reconstruction
        if config.model.use_reconstruction_regularization:
            self.use_regularization = True
            self.max_mask_rate = config.model.reconstruction_regularization.max_mask_rate
        else:
            self.use_regularization = False
            self.max_mask_rate = 0.0
        
        # Even for not using regularization, we still set these parameters for evaluation
        self.regularization_name = config.model.reconstruction_regularization.name
        self.mask_ratio_method = config.model.reconstruction_regularization.mask_ratio_method

        # Policy (Adaptive Masking or Not)
        try:
            tmp = config.model.reconstruction_regularization.use_policy
            self.use_policy = tmp
        except:
            self.use_policy = False
        if self.use_policy:
            self.policy_net = PolicyNet(config, self.feature_extractor.config.hidden_size, 257)

        # Gumbel-Softmax
        self.gumbel_softmax = None
        try:
            if config.model.reconstruction_regularization.use_gumbel_softmax:
                self.gumbel_softmax = config.model.reconstruction_regularization.gumbel_softmax
        except:
            self.gumbel_softmax = None

        # Pairwise training for REINFORCE
        try:
            self.use_pairwise = self.config.model.reconstruction_regularization.policy.use_pairwise
        except:
            self.use_pairwise = False

        # Policy annealing on Actor & Critic
        try:
            tmp = config.model.reconstruction_regularization.policy.annealing
            self.policy_annealing = tmp if tmp.use_annealing else None
        except:
            self.policy_annealing = None
        self.set_policy_annealing_factor(0, config.training.max_train_steps)

        # Softmax temperature annealing
        try:
            tmp = config.model.reconstruction_regularization.policy.temperature
            self.softmax_annealing = tmp if tmp.use_T else None
        except:
            self.softmax_annealing = None            
        self.set_policy_softmax_temperature(0, config.training.max_train_steps)

        # Gaussian smoothing on logits
        try:
            tmp = config.model.reconstruction_regularization.policy.gaussian_smoothing
            self.gaussian_smoothing = tmp if tmp.use_gaussian_smoothing else None
        except:
            self.gaussian_smoothing = None
        self.set_gaussian_smoothing(0, config.training.max_train_steps)
        
        # Set up gaussian sampling
        try:
            tmp = config.model.reconstruction_regularization.policy.gaussian_sampling
            logit_head_type = config.model.reconstruction_regularization.policy.logit_head_type
            self.use_gaussian_sampling = logit_head_type == "gaussian_1"
            self.gaussian_sampling = tmp if self.use_gaussian_sampling else None
        except:
            self.gaussian_sampling = None
            self.use_gaussian_sampling = False
        self.set_gaussian_sampling_sigma(0, config.training.max_train_steps)

        self.num_of_image_tokens = (config.dataset.preprocessing.crop_size // config.model.vq_model.vit_enc_patch_size) ** 2
        self.num_of_latent_tokens = config.model.vq_model.num_latent_tokens

        self.apply(self._init_weights)
        
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

    def set_policy_softmax_temperature(self, global_step: int, max_train_steps: int):
        if self.softmax_annealing is None:
            self.softmax_temperature = 1.0
        else:
            T0 = self.softmax_annealing.T0  # e.g., 0.2
            alpha = self.softmax_annealing.alpha      # e.g., 1.0
            self.softmax_temperature = 1 + T0 * math.exp(- alpha * global_step)

            # progress = global_step / max_train_steps

            # if progress <= start_time:
            #     self.softmax_temperature = start_value
            # elif progress >= end_time:
            #     self.softmax_temperature = end_value
            # else:
            #     # Cosine annealing
            #     normalized_progress = (progress - start_time) / (end_time - start_time)
            #     cosine_decay = 0.5 * (1 + math.cos(math.pi * normalized_progress))
            #     self.softmax_temperature = end_value + (start_value - end_value) * cosine_decay

    def set_gaussian_smoothing(self, global_step: int, max_train_steps: int, start_time=0.0, end_time=0.5, end_value=1.0):
        if self.gaussian_smoothing:
            self.gaussian_kernel_size = self.gaussian_smoothing.kernel_size
            start_value = (self.gaussian_kernel_size - 1) // 2
        else:
            self.gaussian_kernel_size = None
            return

        progress = global_step / max_train_steps

        if progress <= start_time:
            self.gaussian_smoothing.sigma = start_value
        elif progress >= end_time:
            self.gaussian_smoothing.sigma = end_value
        else:
            # Cosine annealing
            normalized_progress = (progress - start_time) / (end_time - start_time)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * normalized_progress))
            self.gaussian_smoothing.sigma = end_value + (start_value - end_value) * cosine_decay

    def set_policy_annealing_factor(self, global_step: int, max_train_steps: int):
        if self.policy_annealing is None:
            self.annealing_factor = 1.0
        else:
            alpha_end = self.policy_annealing.alpha_end # 1
            alpha_start = self.policy_annealing.alpha_start # 0

            progress = global_step / max_train_steps

            if progress < alpha_start:
                self.annealing_factor = 0.0 # QY: Default starting alpha value, no actor loss
            elif progress >= alpha_end:
                self.annealing_factor = 1.0 # QY: Default ending alpha value, emphasize actor loss
            else:
                normalized_progress = (progress - alpha_start) / (alpha_end - alpha_start)
                self.annealing_factor = (math.sin(0.5 * math.pi * normalized_progress) ** 2)

    def set_gaussian_sampling_sigma(self, global_step: int, max_train_steps: int):
        progress = global_step / max_train_steps
        if self.use_gaussian_sampling:
            self.gaussian_sampling_sigma = 0.03 / (0.03 + progress)
        else:
            self.gaussian_sampling_sigma = 1

    def create_key_padding_mask(self, mask_rate):
        """
        Create a key_padding_mask for nn.MultiheadAttention.
        For each sample in the batch, the first N1 tokens are always unmasked,
        while in the second block of N2 tokens, the last int(mask_rate[i] * N2)
        tokens are masked out.

        Args:
            mask_rate (Tensor): shape [B,] with values in [0, 1]
            num_of_image_tokens (int): number of tokens in the first block (always unmasked)
            num_of_latent_tokens (int): number of tokens in the second block

        Returns:
            key_padding_mask (Tensor): Boolean tensor of shape [B, N1+N2] where
                                    True indicates a masked token.
        """
        B = mask_rate.shape[0]
        device = mask_rate.device
        
        mask_first = torch.zeros(B, 1 + self.num_of_image_tokens, dtype=torch.bool, device=device) # Be aware of [cls_token]
        indices = torch.arange(self.num_of_latent_tokens, device=device).unsqueeze(0).expand(B, self.num_of_latent_tokens)
        num_unmasked = (self.num_of_latent_tokens - (mask_rate * self.num_of_latent_tokens).floor()).to(torch.long).unsqueeze(1)
        mask_second = indices >= num_unmasked
        key_padding_mask = torch.cat([mask_first, mask_second], dim=1)
        
        return key_padding_mask

    def encode(self, x, token_features: torch.Tensor, fixed_mask_rate: torch.Tensor = None):
        # Get key padding mask: if fixed mask rate is provided, use it; otherwise, use policy net to get mask rate
        if fixed_mask_rate is not None: # For specific evaluation
            key_padding_mask = self.create_key_padding_mask(fixed_mask_rate)
            output_dict = {}
        elif self.use_policy: # For training and general evaluation
            # in some cases, the policy_net function is vmap
            output_dict = self.policy_net(
                token_features, 
                temperature=self.softmax_temperature, 
                gumbel_softmax=self.gumbel_softmax,
                gaussian_smoothing=self.gaussian_smoothing,
                gaussian_sampling_sigma=self.gaussian_sampling_sigma,
                annealing_factor=self.annealing_factor
            )
            key_padding_mask = self.create_key_padding_mask(output_dict["sampled_mask_rate"]).to(x.device)
        else:
            raise ValueError("Either fixed_mask_rate or policy_net must be provided")

        if self.finetune_decoder:
            with torch.no_grad():  
                self.encoder.eval()
                self.quantize.eval()
                z, _ = self.encoder(
                    pixel_values=x, 
                    latent_tokens=self.latent_tokens,
                    key_padding_mask=key_padding_mask
                )
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
                
        else:
            z, _ = self.encoder(
                pixel_values=x, 
                latent_tokens=self.latent_tokens,
                key_padding_mask=key_padding_mask
            )
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        result_dict.update(output_dict)

        return z_quantized, result_dict

    def get_mask_rate(self, x, decode_mask_rate=0.0):
        device = x.device
        if self.use_regularization and self.training:
            if self.mask_ratio_method == "uniform":
                mask_rate = torch.empty(x.shape[0], device=device).uniform_(0, self.max_mask_rate - 1e-3)
            elif self.mask_ratio_method == "hierarchical":
                values = torch.tensor([i / 16 for i in range(16)], device=device)  # we do not consider zero-token setting
                import math
                upper_bound = math.ceil(self.max_mask_rate * values.shape[0])
                upper_bound = 1 if upper_bound == 0 else upper_bound
                indices = torch.randint(0, upper_bound, (x.shape[0],), device=device)
                mask_rate = values[indices]
            else:
                raise NotImplementedError(f"Unsupported mask ratio method {self.mask_ratio_method}.")
        else:
            mask_rate = torch.tensor(decode_mask_rate, device=device).expand(x.shape[0])

        return mask_rate
    
    def decode(self, z_quantized, decode_mask_rate=0.0):
        if self.training and not self.use_regularization:
            # force decode_mask_rate to be 0 during training for training if not using regularization
            decode_mask_rate = 0.0
        if isinstance(decode_mask_rate, float):
            decode_mask_rate = torch.tensor(decode_mask_rate, device=z_quantized.device).expand(z_quantized.shape[0])

        key_padding_mask = self.create_key_padding_mask(decode_mask_rate)

        if len(decode_mask_rate.shape) == 2:
            assert decode_mask_rate.shape[-1] == z_quantized.shape[-1]
            z_quantized = decode_mask_rate[:, None, None] * z_quantized
        else:
            # mask rate: [B,]
            if self.regularization_name == "matryoshka":
                z_quantized = self.matryoshka_masking(z_quantized, mask_rate=decode_mask_rate)
            elif self.regularization_name == "random":
                raise NotImplementedError(
                    "This training approach has been deprecated.")
                z_quantized = self.random_masking(z_quantized, mask_rate=decode_mask_rate)
            else:
                raise NotImplementedError(f"Unsupported reconstruction regularization {self.reconstruction_regularization}.")
        # z_quantized.shape: [batch_size, token_dim, 1, num_tokens]
        decoded = self.decoder(z_quantized, key_padding_mask=key_padding_mask)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        # decoded.shape: [batch_size, 1024, H, W]
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
    
    def forward(self, x, dino_input=None, fixed_mask_rate_val=0.0, use_fixed_mask_rate=False):
        if not isinstance(fixed_mask_rate_val, float):
            raise ValueError("decode_mask_rate in forward() should be a float")
        

        # QY: If dino_input is not provided, use the original image to form the DINO input
        if dino_input is None:
            print("\033[91mCHECK Not recommended settings: dino_input is None\033[0m")
            dino_input = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
        # Get token features from DINO
        with torch.no_grad():
            token_features = self.feature_extractor(dino_input).last_hidden_state

        # Step 1: MASKED ENCODING
        if self.use_policy and not use_fixed_mask_rate:
            # Use policy net to estimate the mask rate

            if self.use_pairwise and self.training:
                z_quantized, result_dict = self.encode(
                    torch.concat([x,x]), 
                    torch.concat([token_features, token_features])
                )
            else:
                z_quantized, result_dict = self.encode(x, token_features)

            result_dict["annealing_factor"] = self.annealing_factor
            result_dict["softmax_temperature"] = self.softmax_temperature
            forward_mask_rate = result_dict["sampled_mask_rate"]

        else:
            forward_mask_rate = self.get_mask_rate(x, fixed_mask_rate_val)
            z_quantized, result_dict = self.encode(x, token_features, forward_mask_rate)
            result_dict["sampled_mask_rate"] = forward_mask_rate
            result_dict["mask_rate_value"] = forward_mask_rate
        
        # STEP 2: MASKED DECODING
        decoded = self.decode(z_quantized, decode_mask_rate=forward_mask_rate)

        return decoded, result_dict
