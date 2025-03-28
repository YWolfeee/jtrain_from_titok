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
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder, PolicyNet, FlowDecoder
from modeling.quantizer.quantizer import VectorQuantizer
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
import math
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

TORCH_DTYPE=torch.bfloat16


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {
                "channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256,
            }
        )
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25
        )
        # Load pretrained weights
        self.load_state_dict(
            torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True
        )

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


class TiTok(
    BaseModel,
    PyTorchModelHubMixin,
    tags=["arxiv:2406.07550", "image-tokenization"],
    repo_url="https://github.com/bytedance/1d-tokenizer",
    license="apache-2.0",
):
    """Reformulated TiTok model to generalize 1D tokenizer, combining TiTok and FlexTok together. 
    
    Training Pipeline:
    1. pixel_encode(): Encode pixel into 2D patch tokens with Pixel Encoder: 
        TiTok -> identity;
        FlexTok -> VAE;
    2. latent_encode(): Encode 2D patch tokens into 1D discrete latent tokens with Latent Encoder:
        both -> ViT with input tokens as [2D patch tokens, 1D latent tokens];
    3. latent_decode(): Decode 1D latent tokens into 2D patch tokens with Latent Decoder: 
        TiTok -> ViT predict 2D tokens by MaskGiT-VQGAN;
        FlexTok -> ViT predict noise with input as noisy patch tokens;
    4. pixel_decode(): Decode 2D patch tokens into pixel with Pixel Decoder: 
        TiTok -> predict pixel by MaskGiT-VQGAN; 
        FlexTok -> predict pixel by VAE;

    Loss:
    1. TiTok:
        - Stage 1: pixel_encode() -> latent_encode() -> latent_decode() -> ReconstructionLoss_Stage1: CE(predicted_distribution_over_codes, proxy_codes);
        - Stage 2: pixel_encode() -> latent_encode()[freezed] -> latent_decode() -> pixel_decode() -> reconstructed_image -> ReconstructionLoss_Stage2: MSE/Perceptual(reconstructed_image, target_image).
    2. FlexTok:
        - Stage 0 (Skipped): training VAE, in this code base, we use pretrained VAE from FLUX.1-schnell;
        - Stage 1: pixel_encode()[freezed] -> latent_encode() -> latent_decode(flow_training_decode) -> FlowMatchingLoss_Stage1: MSE(predict, flow_target).
    
    Inference Pipeline:
    1. TiTok: exactly the same as Stage 2 training process;
    2. FlexTok: The latent_decode() process is different, it's obtained by sampling with flow matching:
        pixel_encode() -> latent_encode() -> latent_decode(flow_inference_decode) -> pixel_decode() -> reconstructed_image.
        
    """
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        # TODO: Add FSQ
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")

        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError(
                "Only supprot finetune_decoder with vq quantization for now."
            )

        self.from_continuous = config.model.vq_model.get("from_continuous", False)

        # 1. Init
        self.use_encoder_mask = getattr(
            config.model.reconstruction_regularization, "use_encoder_mask", False
        )
        self.num_image_tokens = (
            config.dataset.preprocessing.crop_size
            // config.model.vq_model.vit_enc_patch_size
        ) ** 2
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens

        # 1.1 Init Latent Encoder
        self.latent_encoder = TiTokEncoder(config)
        
        # 1.2 Init Latent Tokens
        scale = self.latent_encoder.width**-0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.latent_encoder.width)
        )

        # 1.3 Init Latent Decoder
        if self.from_continuous:
            self.latent_decoder = FlowDecoder(config)
            self.token_size = self.config.model.vq_model.token_size
            self.null_condition = nn.Parameter(
                torch.zeros(1, self.token_size, 1, self.num_latent_tokens)
            )
            self.null_condition_prob = getattr(
                self.config.model.vq_model, "null_condition_prob", 0.2
            )
            self.num_inference_steps = getattr(
                self.config.model.vq_model, "num_inference_steps", 25
            )
            self.guidance_scale = getattr(
                self.config.model.vq_model, "guidance_scale", 5.0
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0
            )
        else:
            self.latent_decoder = TiTokDecoder(config)

        # 2. Init Reconstruction regularization
        self.use_regularization = getattr(
            config.model, "use_reconstruction_regularization", False
        )
        if self.use_regularization:
            self.max_mask_rate = getattr(
                config.model.reconstruction_regularization, "max_mask_rate", 0.95
            )
        else:
            self.max_mask_rate = 0.0
        self.regularization_name = getattr(
            config.model.reconstruction_regularization, "name", "matryoshka"
        )
        self.mask_ratio_method = getattr(
            config.model.reconstruction_regularization,
            "mask_ratio_method",
            "hierarchical",
        )

        # 3. Init PolicyNet for adaptive masking
        self.use_policy = getattr(
            config.model.reconstruction_regularization, "use_policy", False
        )
        if self.use_policy:
            # 1024: hidden size of dinov2-large; 257: number of tokens of an image + cls_tokens
            self.policy_net = PolicyNet(config, 1024, self.num_latent_tokens)
        try:
            tmp = config.model.reconstruction_regularization.policy.annealing
            self.policy_annealing = tmp if tmp.use_annealing else None
        except:
            self.policy_annealing = None
        self.set_policy_annealing_factor(0, config.training.max_train_steps)

        # 4. Init Weight of Latent Encoder / Latent Decoder / PolicyNet (if exists)
        self.apply(self._init_weights)

        # 5. Init Quantizer
        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
            )
        else:
            raise NotImplementedError

        # 6. Init Feature Extractor
        self.feature_extractor_name = getattr(
            config.model.reconstruction_regularization.policy,
            "feature_extractor_name",
            "facebook/dinov2-large",
        )
        self.feature_extractor = AutoModel.from_pretrained(self.feature_extractor_name)
        self.feature_extractor = self.feature_extractor.to(dtype=TORCH_DTYPE)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(
            False
        )  # Output of feature extractor: [B, 257, 768] for base; [B, 257, 1024] for large

        # 7. Init PixelEncoder, PixelQuantizer, PixelDecoder
        if self.from_continuous:
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                subfolder="vae",
                torch_dtype=TORCH_DTYPE,
            )
            self.vae.eval()
            self.vae.requires_grad_(False)

        if self.finetune_decoder:
            self.latent_tokens.requires_grad_(False)
            self.latent_encoder.eval()
            self.latent_encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            if self.use_policy:
                self.policy_net.eval()
                self.policy_net.requires_grad_(False)

            if not self.from_continuous:
                self.pixel_quantize = Pixel_Quantizer(
                    num_embeddings=1024, embedding_dim=256, commitment_cost=0.25
                )
                self.pixel_decoder = Pixel_Decoder(
                    OmegaConf.create(
                        {
                            "channel_mult": [1, 1, 2, 2, 4],
                            "num_resolutions": 5,
                            "dropout": 0.0,
                            "hidden_channels": 128,
                            "num_channels": 3,
                            "num_res_blocks": 2,
                            "resolution": 256,
                            "z_channels": 256,
                        }
                    )
                )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory.

        Args:
            save_directory (Path): Directory to save weights and config
        """
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, "w") as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module (torch.nn.Module): Module to initialize
        """
        if (
            isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
        ):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_guidance_scale(self, guidance_scale: float):
        self.guidance_scale = guidance_scale
    
    def set_max_mask_rate(self, global_step: int, max_train_steps: int):
        """Set maximum mask rate.

        Args:
            global_step (int): Current global step
            max_train_steps (int): Maximum training steps
        """
        annealing = self.config.model.reconstruction_regularization.annealing
        is_increasing = annealing.is_increasing
        time_start = annealing.time_start * max_train_steps
        time_end = annealing.time_end * max_train_steps
        alpha = (global_step - time_start) / (time_end - time_start)
        end_mask_rate = (
            self.config.model.reconstruction_regularization.max_mask_rate
            if is_increasing
            else 0.0
        )
        start_mask_rate = (
            0.0
            if is_increasing
            else self.config.model.reconstruction_regularization.max_mask_rate
        )
        if global_step < time_start:
            return start_mask_rate
        elif global_step > time_end:
            return end_mask_rate
        else:
            return alpha * end_mask_rate + (1 - alpha) * start_mask_rate

    def set_policy_annealing_factor(self, global_step: int, max_train_steps: int):
        """Set the annealing factor inside the policy.

        Args:
            global_step (int): Current global step
            max_train_steps (int): Maximum training steps
        """
        if self.policy_annealing is None:
            self.annealing_factor = 1.0
        else:
            alpha_end = self.policy_annealing.get(
                "alpha_end", 1.0
            )  # When to end annealing
            alpha_start = self.policy_annealing.get(
                "alpha_start", 0.0
            )  # When to start annealing

            progress = global_step / max_train_steps
            if progress < alpha_start:
                self.annealing_factor = 0.0  # QY: Default starting value
            elif progress >= alpha_end:
                self.annealing_factor = 1.0  # QY: Default ending value
            else:
                normalized_progress = (progress - alpha_start) / (
                    alpha_end - alpha_start
                )
                self.annealing_factor = (
                    math.sin(0.5 * math.pi * normalized_progress) ** 2
                )

    def _create_key_padding_mask(self, mask_rate):
        """Create a key_padding_mask for nn.MultiheadAttention.
        For each sample in the batch, the first N1 tokens are always unmasked,
        while in the second block of N2 tokens, the last int(mask_rate[i] * N2)
        tokens are masked out.

        Args:
            mask_rate (Tensor): Shape [B,] with values in [0, 1]

        Returns:
            key_padding_mask (Tensor): Boolean tensor of shape [B, N1+N2] where True indicates a masked token
        """
        B = mask_rate.shape[0]
        device = mask_rate.device

        mask_first = torch.zeros(
            B, 1 + self.num_image_tokens, dtype=torch.bool, device=device
        )  # Be aware of [cls_token]
        indices = (
            torch.arange(self.num_latent_tokens, device=device)
            .unsqueeze(0)
            .expand(B, self.num_latent_tokens)
        )
        num_unmasked = (
            (self.num_latent_tokens - (mask_rate * self.num_latent_tokens).floor())
            .to(torch.long)
            .unsqueeze(1)
        )
        mask_second = indices >= num_unmasked
        key_padding_mask = torch.cat([mask_first, mask_second], dim=1)

        return key_padding_mask

    def _create_encoder_attn_mask(self):
        """Create encoder attention mask for nn.MultiheadAttention.
        The image tokens are always attended to each other but not to the latent tokens,
        while the latent tokens are enforced causality.
        E.g., num_image_tokens=3, num_latent_tokens=2, the mask is:
        [[False, False, False, True, True],
        [False, False, False, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False]]
        Here False means attended, while True means not attended.

        Returns:
            attn_mask (Tensor): Attention mask tensor
        """
        full_seq_len = 1 + self.num_image_tokens + self.num_latent_tokens
        attn_mask = torch.tril(torch.ones(full_seq_len, full_seq_len, dtype=torch.bool))
        attn_mask[: self.num_image_tokens, : self.num_image_tokens] = True
        return ~attn_mask

    @torch.no_grad()
    def pixel_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to 2D latent tokens using pixel encoder (a VAE) if from_continuous is True,
        otherwise returns input unchanged under the setting of TiTok.

        Args:
            x (torch.Tensor): Input images tensor

        Returns:
            torch.Tensor: 2D tokens if from_continuous is True, otherwise returns image
        """
        if self.from_continuous:
            self.vae.eval()
            x = (x - 0.5) / 0.5 # Rescale from [0, 1] to [-1, 1] for VAE input
            latent_dist = self.vae.encode(x).latent_dist
            # shift_factor: 0.1159; scaling_factor: 0.3611
            x = (latent_dist.sample() - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return x

    def latent_encode(
        self,
        x,
        dino_feature: torch.Tensor,
        fixed_mask_rate: torch.Tensor = None,
        vae_results: dict = None,
    ):
        """Encode 2D latents (if from_continuous is True) / images (if from_continuous is False) to 1D tokens.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W). Either 2D latents from VAE or raw images
            dino_feature (torch.Tensor): Token features from DINO of shape (B, 257, 1024) (obtained from large)
            fixed_mask_rate (torch.Tensor, optional): Fixed mask rate tensor of shape (B,). Defaults to None
            vae_results (dict, optional): Results dictionary from VAE encoding. Defaults to None

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing:
                - z_quantized (torch.Tensor): Quantized latent tokens of shape (B, C, H, W)
                - result_dict (dict): Dictionary containing quantization and policy results
        """
        # 1. Get key padding mask based on mask rate
        if fixed_mask_rate is not None:
            key_padding_mask = self._create_key_padding_mask(fixed_mask_rate)
            output_dict = {}
            encode_mask_rate = fixed_mask_rate
        elif self.use_policy:
            output_dict = self.policy_net(
                dino_feature,
                annealing_factor=self.annealing_factor,
                vae_results=vae_results,
            )
            key_padding_mask = self._create_key_padding_mask(
                output_dict["sampled_mask_rate"]
            ).to(x.device)
            encode_mask_rate = output_dict["sampled_mask_rate"]
        else:
            raise ValueError("Either fixed_mask_rate or policy_net must be provided")

        # 2. Get encoder attention mask (causality on latent tokens)
        encoder_mask = (
            self._create_encoder_attn_mask().to(x.device)
            if self.use_encoder_mask
            else None
        )

        # 3. Encode to 1D tokens
        if self.finetune_decoder:
            with torch.no_grad():
                self.latent_encoder.eval()
                self.quantize.eval()
                z, _ = self.latent_encoder(
                    pixel_values=x,
                    latent_tokens=self.latent_tokens,
                    key_padding_mask=key_padding_mask,
                    attn_mask=encoder_mask,
                )
                z_quantized, result_dict = self.quantize(z, mask_rate=encode_mask_rate)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z, _ = self.latent_encoder(
                pixel_values=x,
                latent_tokens=self.latent_tokens,
                key_padding_mask=key_padding_mask,
                attn_mask=encoder_mask,
            )
            z_quantized, result_dict = self.quantize(z, mask_rate=encode_mask_rate)

        result_dict.update(output_dict)
        return z_quantized, result_dict

    def encode(
        self,
        x,
        dino_feature: torch.Tensor,
        fixed_mask_rate: torch.Tensor = None,
        vae_results: dict = None,
    ):
        """Encode input images by pixel_encode() then latent_encode().

        Args:
            x (Tensor): Input images
            dino_feature (Tensor): Token features from feature extractor
            fixed_mask_rate (Tensor, optional): Fixed mask rate, the mask rate will not rely on policy net if specified
            vae_results (dict, optional): VAE results including the ELBO. Defaults to None

        Returns:
            Tuple[Tensor, dict]: Quantized features and encoding results
        """
        latent_input = self.pixel_encode(x)
        z_quantized, result_dict = self.latent_encode(
            latent_input, dino_feature, fixed_mask_rate, vae_results
        )
        result_dict["vae_latent"] = latent_input
        return z_quantized, result_dict

    def latent_decode(self, z_quantized, decode_mask_rate=0.0, extra_dict={}, to_pixel=False):
        """Decode 1D quantized features to 2D quantized tokens (if from_continuous is False, from MaskGiT-VQGAN)
        or 2D continuous tokens (if from_continuous is True, from FLUX VAE).

        Args:
            z_quantized (Tensor): Quantized features
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0
            extra_dict: Extra dictionary for latent decoder, include vae_latent if using flow matching

        Returns:
            Tensor: Decoded images
        """
        if self.training and not self.use_regularization:
            # force decode_mask_rate to be 0 during training if not using regularization
            decode_mask_rate = 0.0
        if isinstance(decode_mask_rate, float):
            decode_mask_rate = torch.tensor(
                decode_mask_rate, device=z_quantized.device
            ).expand(z_quantized.shape[0])

        key_padding_mask = self._create_key_padding_mask(decode_mask_rate)

        latent_decode_dict = {}
        # z_quantized.shape: [batch_size, token_dim, 1, num_tokens]
        if not self.from_continuous:
            decoded = self.latent_decoder(
                z_quantized, key_padding_mask=key_padding_mask
            )
        elif self.from_continuous and not to_pixel:
            decoded, flow_target, repa_feature = self._flow_training_decode(
                z_quantized, extra_dict["vae_latent"], key_padding_mask=key_padding_mask
            )
            latent_decode_dict["flow_target"] = flow_target
            latent_decode_dict["repa_feature"] = repa_feature
        elif self.from_continuous and to_pixel:
            decoded = self._flow_inference_decode(
                z_quantized, key_padding_mask=key_padding_mask
            )

        return decoded, latent_decode_dict
    
    def _get_sigmas(
        self, 
        indices: torch.Tensor, 
        n_dim: int = 4, 
        device = None,
        dtype: torch.dtype = TORCH_DTYPE
    ) -> torch.Tensor:
        """Retrieves sigma values corresponding to given timesteps from the scheduler.
        
        This function is reproduced from diffusers flux training pipeline.
        Source: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py
        
        Args:
            timesteps: A tensor containing timestep values, (B,)
            n_dim: The number of dimensions for the output sigma tensor. Defaults to 4 because the vae latent is [B, 16, 32, 32]
            dtype: The data type for the sigma values. Defaults to TORCH_DTYPE.
            
        Returns:
            A tensor containing sigma values with shape expanded to n_dim: (B, 1, 1)
        """
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        sigma = sigmas[indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def _compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
        device = "cpu"
    ):
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device=device)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u

    def _flow_training_decode(
        self, z_quantized, vae_latent, key_padding_mask=None
    ):
        """Perform flow matching training with Classifier-Free Guidance.

        Args:
            z_quantized (torch.Tensor): Quantized features of shape [batch_size, token_dim, 1, num_tokens]
            vae_latent (torch.Tensor): VAE latent features of shape [batch_size, 16, 32, 32]
            key_padding_mask (torch.Tensor, optional): Key padding mask for attention. Defaults to None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - predict_noise (torch.Tensor): Predicted noise
                - noise (torch.Tensor): Original noise added to the latent
        """
        self.scheduler.set_timesteps(num_inference_steps=1000, device=z_quantized.device)
        batch_size = z_quantized.shape[0]

        # To enable CFG, replace z_quantized with null_condition with a probability of self.null_condition_prob
        random_drop_mask = (
            torch.rand(batch_size, device=z_quantized.device)
            >= self.null_condition_prob
        ).to(dtype=z_quantized.dtype)
        mask_expanded = random_drop_mask.view(batch_size, 1, 1, 1).to(
            z_quantized.device
        )  # [batch_size, 1, 1, 1]
        
        null_condition_expanded = self.null_condition.expand(batch_size, -1, -1, -1)
        z_quantized = (
            mask_expanded * z_quantized + (1 - mask_expanded) * null_condition_expanded
        )  # [batch_size, token_dim, 1, num_tokens]

        # Sample random timestep indices for each item in the batch, use SD3 training scheme
        u = self._compute_density_for_timestep_sampling(
            weighting_scheme="mode",
            batch_size=batch_size,
            mode_scale=0.25,
            device=z_quantized.device
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        # Handle the edge case where u = 1
        indices = torch.clamp(indices, min=0, max=self.scheduler.config.num_train_timesteps - 1)
        timesteps = self.scheduler.timesteps[indices].to(device=z_quantized.device)
        sigmas = self._get_sigmas(indices, n_dim=vae_latent.ndim, device=z_quantized.device, dtype=z_quantized.dtype)
        noise = torch.randn_like(vae_latent)
        noisy_vae_latent = (1.0 - sigmas) * vae_latent + sigmas * noise

        # Flow prediction
        predict, repa_feature = self.latent_decoder(
            z_quantized, noisy_vae_latent, timesteps, key_padding_mask=key_padding_mask
        )
        target = noise - vae_latent

        return predict, target, repa_feature

    @torch.no_grad()
    def _flow_inference_decode(self, z_quantized, key_padding_mask=None):
        """Perform flow matching inference with Classifier-Free Guidance.

        Args:
            z_quantized (torch.Tensor): Quantized features of shape [batch_size, token_dim, 1, num_tokens]
            key_padding_mask (torch.Tensor, optional): Key padding mask for attention. Defaults to None

        Returns:
            torch.Tensor: Generated latents of shape [batch_size, 16, 32, 32]
        """
        # Flow Matching Inference with Classifier-Free Guidance
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=z_quantized.device)
        batch_size = z_quantized.shape[0]
        latent_shape = (batch_size, 16, 32, 32)  # VAE latent shape

        # Get both conditional and unconditional latents
        null_condition_expanded = self.null_condition.expand(batch_size, -1, -1, -1)

        # Start with random noise
        latents = torch.randn(latent_shape, device=z_quantized.device)

        # Denoising loop
        for t in self.scheduler.timesteps:
            timestep = t.expand(batch_size)

            # Concatenate conditional and unconditional inputs to process in a single forward pass
            combined_condition = torch.cat([z_quantized, null_condition_expanded], dim=0)
            combined_latents = torch.cat([latents, latents], dim=0)
            combined_timestep = torch.cat([timestep, timestep], dim=0)
            
            # Create combined padding mask if needed
            combined_key_padding_mask = None
            if key_padding_mask is not None:
                combined_key_padding_mask = torch.cat([key_padding_mask, key_padding_mask], dim=0)
            
            # Single forward pass for both conditional and unconditional predictions
            combined_pred, _ = self.latent_decoder(
                combined_condition, 
                combined_latents, 
                combined_timestep, 
                key_padding_mask=combined_key_padding_mask
            )
            
            # Split the predictions back into conditional and unconditional
            cond_pred, uncond_pred = torch.chunk(combined_pred, 2, dim=0)
            
            # Apply classifier-free guidance
            final_pred = uncond_pred + self.guidance_scale * (cond_pred - uncond_pred)

            # Update latents with scheduler step
            latents = self.scheduler.step(
                model_output=final_pred, timestep=t, sample=latents
            ).prev_sample

        return latents

    def pixel_decode(self, decoded: torch.Tensor) -> torch.Tensor:
        """Decode 2D tokens to pixels using either continuous VAE decoder or discrete VQ decoder.

        Args:
            decoded (torch.Tensor): Decoded tokens from latent decoder

        Returns:
            torch.Tensor: Decoded pixel values
        """
        if self.from_continuous:
            latents_for_decode = decoded / self.vae.config.scaling_factor + self.vae.config.shift_factor
            decoded = self.vae.decode(latents_for_decode).sample
        else:
            quantized_states = torch.einsum(
                "nchw,cd->ndhw",
                decoded.softmax(1),
                self.pixel_quantize.embedding.weight,
            )
            decoded = self.pixel_decoder(quantized_states)
        return decoded

    def decode(self, z_quantized, decode_mask_rate=0.0, extra_dict={}, to_pixel=False):
        """Decode quantized features.

        Args:
            z_quantized (Tensor): Quantized features
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0

        Returns:
            Tensor: Decoded images
        """
        decoded, latent_decode_dict = self.latent_decode(
            z_quantized, decode_mask_rate, extra_dict, to_pixel
        )
        if self.finetune_decoder or to_pixel:
            decoded = self.pixel_decode(decoded)
        return decoded, latent_decode_dict

    @torch.no_grad()
    def encode_tokens(
        self,
        images,
        dino_input=None,
        fixed_mask_rate_val=0.0,
        use_fixed_mask_rate=False,
        vae_results=None,
    ):
        """Encode images to tokens. This is for reference.

        Args:
            images (Tensor): Input images
            dino_input (Tensor, optional): DINO input. Defaults to None
            fixed_mask_rate_val (float, optional): Fixed mask rate value. Defaults to 0.0
            use_fixed_mask_rate (bool, optional): Whether to use fixed mask rate. Defaults to False
            vae_results (dict, optional): VAE results. Defaults to None

        Returns:
            Tuple[Tensor, Tensor]: Full tokens and mask rate
        """
        if not isinstance(fixed_mask_rate_val, float):
            raise ValueError("decode_mask_rate in forward() should be a float")

        # QY: If dino_input is not provided, use the original image to form the DINO input
        if dino_input is None:
            dino_input = torch.nn.functional.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Get token features from DINO
        with torch.no_grad():
            dino_feature = self.feature_extractor(dino_input).last_hidden_state

        if self.use_policy and not use_fixed_mask_rate:
            # Use policy net to estimate the mask rate
            _, encode_dict = self.encode(
                images, dino_feature, vae_results=vae_results
            )
        else:
            forward_mask_rate = self.get_mask_rate(images, fixed_mask_rate_val)
            _, encode_dict = self.encode(images, dino_feature, forward_mask_rate)
        full_tokens = encode_dict["min_encoding_indices"].reshape(images.shape[0], -1)
        mask_rate = encode_dict["sampled_mask_rate"].reshape(images.shape[0], -1)
        return full_tokens, mask_rate

    @torch.no_grad()
    def decode_tokens(self, tokens, decode_mask_rate=0.0):
        """Decode tokens to images. This is for inference.

        Args:
            tokens (Tensor): Input tokens
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0

        Returns:
            Tensor: Decoded images
        """
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape  # B x N1
        # padding after tokens such that shape at dim=1 is self.num_latent_tokens
        if seq_len < self.num_latent_tokens:
            # no specific meanning of id=0, since it will later be ignored by the attention mask
            padding = torch.zeros(
                batch,
                self.num_latent_tokens - seq_len,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, padding], dim=1)
        z_quantized = self.quantize.get_codebook_entry(tokens.reshape(-1)).reshape(
            batch, 1, seq_len, -1
        )
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
        decode_mask_rate = 1 - seq_len / self.num_latent_tokens
        decoded = self.decode(z_quantized, decode_mask_rate=decode_mask_rate)
        return decoded

    def get_mask_rate(self, x, decode_mask_rate=0.0):
        """Get mask rate based on training settings.

        Args:
            x (Tensor): Input tensor
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0

        Returns:
            Tensor: Mask rate tensor
        """
        device = x.device
        if self.use_regularization and self.training:
            if self.mask_ratio_method == "uniform":
                mask_rate = torch.empty(x.shape[0], device=device).uniform_(
                    0, self.max_mask_rate - 1e-3
                )
            elif self.mask_ratio_method == "hierarchical":
                values = torch.tensor(
                    [i / 16 for i in range(16)], device=device
                )  # we do not consider zero-token setting
                upper_bound = math.ceil(self.max_mask_rate * values.shape[0])
                upper_bound = 1 if upper_bound == 0 else upper_bound
                indices = torch.randint(0, upper_bound, (x.shape[0],), device=device)
                mask_rate = values[indices]
            else:
                raise NotImplementedError(
                    f"Unsupported mask ratio method {self.mask_ratio_method}."
                )
        else:
            mask_rate = torch.tensor(decode_mask_rate, device=device).expand(x.shape[0])

        return mask_rate

    def forward(
        self,
        x,
        dino_input=None,
        fixed_mask_rate_val=0.0,
        use_fixed_mask_rate=False,
        vae_results=None,
        to_pixel=False
    ):
        """Forward pass.

        Args:
            x (Tensor): Input images
            dino_input (Tensor, optional): DINO input. Defaults to None
            fixed_mask_rate_val (float, optional): Fixed mask rate value. Defaults to 0.0
            use_fixed_mask_rate (bool, optional): Whether to use fixed mask rate. Defaults to False
            vae_results (dict, optional): VAE results. Defaults to None

        Returns:
            Tuple[Tensor, dict]: Decoded images and results dictionary
        """
        if not isinstance(fixed_mask_rate_val, float):
            raise ValueError("decode_mask_rate in forward() should be a float")

        # 0: Get token features from DINO
        if dino_input is None:
            dino_input = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            dino_feature = self.feature_extractor(dino_input).last_hidden_state

        # 1: MASKED ENCODING
        if self.use_policy and not use_fixed_mask_rate:
            # Use policy net to estimate the mask rate
            z_quantized, result_dict = self.encode(
                x, dino_feature, vae_results=vae_results
            )
            result_dict["annealing_factor"] = self.annealing_factor
            forward_mask_rate = result_dict["sampled_mask_rate"]
        else:
            forward_mask_rate = self.get_mask_rate(x, fixed_mask_rate_val)
            z_quantized, result_dict = self.encode(x, dino_feature, forward_mask_rate)
            result_dict["sampled_mask_rate"] = forward_mask_rate
            result_dict["mask_rate_value"] = forward_mask_rate

        # 2: MASKED DECODING
        decoded, latent_decode_dict = self.decode(
            z_quantized, decode_mask_rate=forward_mask_rate, extra_dict=result_dict, to_pixel=to_pixel
        )
        result_dict.update(latent_decode_dict)
        result_dict["dino_feature"] = dino_feature

        return decoded, result_dict
