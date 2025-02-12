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

        self.policy_net = None
        if self.config.model.reconstruction_regularization.use_policy:
            self.policy_net = PolicyNet(config, self.encoder.width)
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
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
        
        elif self.freeze_decoder:
            self.decoder.eval()
            self.decoder.requires_grad_(False)
        elif self.freeze_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            
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

        try:
            tmp = config.model.reconstruction_regularization.use_policy
            self.use_policy = tmp
        except:
            self.use_policy = False

        self.gumbel_softmax = None
        try:
            if config.model.reconstruction_regularization.use_gumbel_softmax:
                self.gumbel_softmax = config.model.reconstruction_regularization.gumbel_softmax
        except:
            self.gumbel_softmax = None

        try:
            tmp = config.model.reconstruction_regularization.policy.annealing
            self.policy_annealing = tmp if tmp.use_annealing else None
        except:
            self.policy_annealing = None
        self.set_policy_annealing_factor(0, config.training.max_train_steps)

        try:
            tmp = config.model.reconstruction_regularization.policy.temperature
            self.softmax_annealing = tmp if tmp.use_T else None
        except:
            self.softmax_annealing = None            
        self.set_policy_softmax_temperature(0, config.training.max_train_steps)

        try:
            tmp = config.model.reconstruction_regularization.policy.gaussian_smoothing
            self.gaussian_smoothing = tmp if tmp.use_gaussian_smoothing else None
        except:
            self.gaussian_smoothing = None
        self.set_gaussian_smoothing(0, config.training.max_train_steps)
        
        try:
            tmp = config.model.reconstruction_regularization.policy.training_regime
            self.training_regime = tmp if tmp.use_training_regime else None
        except:
            self.training_regime = None
        self.set_training_regime(0, config.training.max_train_steps)
        
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
            self.gaussian_smoothing.sigma = None
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

    def set_training_regime(self, global_step: int, max_train_steps: int):
        if self.training_regime is None:
            return

        training_regime = self.training_regime
        progress = global_step / max_train_steps

        print("\033[91mmax_train_steps: ", max_train_steps, "\033[0m")
        print("\033[91mglobal_step: ", global_step, "\033[0m")
        print("\033[91mfirst_start: ", training_regime.first_start, "\033[0m")
        print("\033[91msecond_start: ", training_regime.second_start, "\033[0m")
        
        assert training_regime.name in ["encoder_then_router_and_decoder", "decoder_then_router_and_encoder"]
        if training_regime.name == "encoder_then_router_and_decoder":
            if progress >= training_regime.first_start:
                # Freeze decoder and policy_net, encoder and latent tokens and quantizer are trainable
                self.latent_tokens.requires_grad_(True)
                self.encoder.train()
                self.encoder.requires_grad_(True)
                self.quantize.train()
                self.quantize.requires_grad_(True)
                self.policy_net.eval()
                self.policy_net.requires_grad_(False)
                self.decoder.eval()
                self.decoder.requires_grad_(False)
            elif progress >= training_regime.second_start:
                # Freeze encoder and latent tokens and quantizer, decoder and policy_net are trainable
                self.latent_tokens.requires_grad_(False)
                self.encoder.eval()
                self.encoder.requires_grad_(False)
                self.quantize.eval()
                self.quantize.requires_grad_(False)
                self.policy_net.train()
                self.policy_net.requires_grad_(True)
                self.decoder.train()
                self.decoder.requires_grad_(True)
            else:
                # All are trainable
                self.latent_tokens.requires_grad_(True)
                self.encoder.train()
                self.encoder.requires_grad_(True)
                self.quantize.train()
                self.quantize.requires_grad_(True)
                self.policy_net.train()
                self.policy_net.requires_grad_(True)
                self.decoder.train()
                self.decoder.requires_grad_(True)
        elif training_regime.name == "decoder_then_router_and_encoder":
            if progress >= training_regime.first_start:
                # Freeze encoder and latent tokens and quantizer and policy_net, decoder is trainable
                self.latent_tokens.requires_grad_(False)
                self.encoder.eval()
                self.encoder.requires_grad_(False)
                self.quantize.eval()
                self.quantize.requires_grad_(False)
                self.policy_net.eval()
                self.policy_net.requires_grad_(False)
                self.decoder.train()
                self.decoder.requires_grad_(True)
            elif progress >= training_regime.second_start:
                # Freeze decoder, encoder and latent tokens and quantizer and policy_net are trainable
                self.latent_tokens.requires_grad_(True)
                self.encoder.train()
                self.encoder.requires_grad_(True)
                self.quantize.train()
                self.quantize.requires_grad_(True)
                self.policy_net.train()
                self.policy_net.requires_grad_(True)
                self.decoder.eval()
                self.decoder.requires_grad_(False)
            else:
                # All are trainable
                self.latent_tokens.requires_grad_(True)
                self.encoder.train()
                self.encoder.requires_grad_(True)
                self.quantize.train()
                self.quantize.requires_grad_(True)
                self.policy_net.train()
                self.policy_net.requires_grad_(True)
                self.decoder.train()
                self.decoder.requires_grad_(True)
        else:
            raise NotImplementedError(f"Unsupported training regime {training_regime.name}.")

    def encode(self, x, policy_net: PolicyNet = None, drop_p=0.0):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z, z_embedding = self.encoder(
                    pixel_values=x, 
                    latent_tokens=self.latent_tokens,
                )
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
                if policy_net:
                    output_dict = policy_net(
                        z_embedding, 
                        temperature=self.softmax_temperature, 
                        gumbel_softmax=self.gumbel_softmax,
                        gaussian_smoothing=self.gaussian_smoothing,
                        annealing_factor=self.annealing_factor,
                        )
        else:
            z, z_embedding = self.encoder(
                pixel_values=x, 
                latent_tokens=self.latent_tokens,
                )
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors
            if policy_net:
                output_dict = policy_net(
                    z_embedding, 
                    temperature=self.softmax_temperature, 
                    gumbel_softmax=self.gumbel_softmax,
                    gaussian_smoothing=self.gaussian_smoothing,
                    annealing_factor=self.annealing_factor,
                    )

        if policy_net:
            result_dict.update(output_dict)

        return z_quantized, result_dict

    def get_mask_rate(self, z_quantized, decode_mask_rate=0.0):
        device = z_quantized.device
        if self.use_regularization and self.training:
            if self.mask_ratio_method == "uniform":
                mask_rate = torch.empty(z_quantized.shape[0], device=device).uniform_(0, self.max_mask_rate - 1e-3)
            elif self.mask_ratio_method == "hierarchical":
                values = torch.tensor([i / 16 for i in range(16)], device=device)  # we do not consider zero-token setting
                import math
                upper_bound = math.ceil(self.max_mask_rate * values.shape[0])
                upper_bound = 1 if upper_bound == 0 else upper_bound
                indices = torch.randint(0, upper_bound, (z_quantized.shape[0],), device=device)
                mask_rate = values[indices]
            else:
                raise NotImplementedError(f"Unsupported mask ratio method {self.mask_ratio_method}.")
        else:
            mask_rate = torch.tensor(decode_mask_rate, device=device).expand(z_quantized.shape[0])

        return mask_rate
    
    def decode(self, z_quantized, decode_mask_rate=0.0):
        if self.training and not self.use_regularization:
            # force decode_mask_rate to be 0 during training for training if not using regularization
            decode_mask_rate = 0.0
        if isinstance(decode_mask_rate, float):
            decode_mask_rate = torch.tensor(decode_mask_rate, device=z_quantized.device).expand(z_quantized.shape[0])

        if len(decode_mask_rate.shape) == 2:
            assert decode_mask_rate.shape[-1] == z_quantized.shape[-1]
            z_quantized = decode_mask_rate[:, None, None] * z_quantized
        else:
            # mask rate is a tensor with shape (batch_size,)
            # values could be identical inside
            if self.regularization_name == "matryoshka":
                z_quantized = self.matryoshka_masking(z_quantized, mask_rate=decode_mask_rate)
            elif self.regularization_name == "random":
                raise NotImplementedError(
                    "This training approach has been deprecated.")
                z_quantized = self.random_masking(z_quantized, mask_rate=decode_mask_rate)
            else:
                raise NotImplementedError(f"Unsupported reconstruction regularization {self.reconstruction_regularization}.")
        # z_quantized.shape: [batch_size, token_dim, 1, num_tokens]
        decoded = self.decoder(z_quantized)
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
    
    def forward(self, x, decode_mask_rate=0.0, fixed_mask_rate=False):
        if not isinstance(decode_mask_rate, float):
            raise ValueError("decode_mask_rate in forward() should be a float")
        
        # Step 1: ENCODE
        z_quantized, result_dict = self.encode(x, policy_net = self.policy_net)

        # Step 2: MASKING
        if self.config.model.reconstruction_regularization.use_policy and not fixed_mask_rate:
            # if using policy, instead of using random mask rate for training, we use the policy to estimate the mask rate
            # Notice that this process has been put in the self.encode function

            # add additional parameters for printing
            result_dict["annealing_factor"] = self.annealing_factor
            result_dict["softmax_temperature"] = self.softmax_temperature
            sampled_mask_rate = result_dict["sampled_mask_rate"]

        else:
            sampled_mask_rate= self.get_mask_rate(z_quantized, decode_mask_rate)
            result_dict["sampled_mask_rate"] = sampled_mask_rate
            result_dict["mask_rate_value"] = sampled_mask_rate
        
        # STEP 3: DECODE
        decoded = self.decode(z_quantized, decode_mask_rate=sampled_mask_rate)
        # DEBUG: If use self-distill, the decode_mask_rate should be used to determine which part corresponds to ground truth (if some is less than 1/16)
        #        The self-distilliated codes are later used to compute the loss, we detach them to avoid back-propagation on decoder twice
        #        The decode_mask_rate is correct in dry_run
        if self.config.losses.use_self_distilliation:
            result_dict["decode_mask_rate"] = sampled_mask_rate
            result_dict["self_distilliated_codes"] = self.decode(z_quantized, torch.maximum(torch.zeros_like(sampled_mask_rate), sampled_mask_rate - 1/16)).detach()
        return decoded, result_dict
