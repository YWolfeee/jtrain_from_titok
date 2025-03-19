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
            raise ValueError(
                "Only supprot finetune_decoder with vq quantization for now."
            )

        # 1. Init Encoder / Decoder / latent_tokens
        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        self.use_encoder_mask = getattr(
            config.model.reconstruction_regularization, "use_encoder_mask", False
        )
        self.num_image_tokens = (
            config.dataset.preprocessing.crop_size
            // config.model.vq_model.vit_enc_patch_size
        ) ** 2
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width**-0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width)
        )

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
            # 768: hidden size of dinov2-base; 257: number of tokens of an image + cls_tokens
            self.policy_net = PolicyNet(config, 768, 257)
        try:
            tmp = config.model.reconstruction_regularization.policy.annealing
            self.policy_annealing = tmp if tmp.use_annealing else None
        except:
            self.policy_annealing = None
        self.set_policy_annealing_factor(0, config.training.max_train_steps)

        # 4. Init Weight of Encoder / Decoder / PolicyNet (if exists)
        self.apply(self._init_weights)

        # 5. Init Quantizer
        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
            )
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

        # 6. Init Feature Extractor
        self.feature_extractor_name = getattr(
            config.model.reconstruction_regularization.policy,
            "feature_extractor_name",
            "facebook/dinov2-base",
        )
        self.feature_extractor = AutoModel.from_pretrained(self.feature_extractor_name)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(
            False
        )  # Output of feature extractor: [B, 257, 768]

        # 7.If fine-tuning decoder, freeze encoder/quantizer/latent tokens and add (PixelQuantizer, PixelDecoder)
        if self.finetune_decoder:
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            if self.use_policy:
                self.policy_net.eval()
                self.policy_net.requires_grad_(False)

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

    def create_key_padding_mask(self, mask_rate):
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

    def create_encoder_attn_mask(self):
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

    def encode(
        self,
        x,
        token_features: torch.Tensor,
        fixed_mask_rate: torch.Tensor = None,
        vae_results: dict = None,
    ):
        """Encode input images.

        Args:
            x (Tensor): Input images
            token_features (Tensor): Token features from feature extractor
            fixed_mask_rate (Tensor, optional): Fixed mask rate, the mask rate will not rely on policy net if specified
            vae_results (dict, optional): VAE results including the ELBO. Defaults to None

        Returns:
            Tuple[Tensor, dict]: Quantized features and encoding results
        """
        # Get key padding mask
        if fixed_mask_rate is not None:
            # Used for evaluation with specified fixed mask rate (not using policy net)
            key_padding_mask = self.create_key_padding_mask(fixed_mask_rate)
            output_dict = {}
            encode_mask_rate = fixed_mask_rate
        elif self.use_policy:
            output_dict = self.policy_net(
                token_features,
                annealing_factor=self.annealing_factor,
                vae_results=vae_results,
            )
            key_padding_mask = self.create_key_padding_mask(
                output_dict["sampled_mask_rate"]
            ).to(x.device)
            encode_mask_rate = output_dict["sampled_mask_rate"]
        else:
            raise ValueError("Either fixed_mask_rate or policy_net must be provided")

        encoder_mask = (
            self.create_encoder_attn_mask().to(x.device)
            if self.use_encoder_mask
            else None
        )
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z, _ = self.encoder(
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
            z, _ = self.encoder(
                pixel_values=x,
                latent_tokens=self.latent_tokens,
                key_padding_mask=key_padding_mask,
                attn_mask=encoder_mask,
            )
            z_quantized, result_dict = self.quantize(z, mask_rate=encode_mask_rate)

        result_dict.update(output_dict)

        return z_quantized, result_dict

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

    def decode(self, z_quantized, decode_mask_rate=0.0):
        """Decode quantized features.

        Args:
            z_quantized (Tensor): Quantized features
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0

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

        key_padding_mask = self.create_key_padding_mask(decode_mask_rate)

        if len(decode_mask_rate.shape) == 2:
            # decode_mask_rate: [B, N]
            assert decode_mask_rate.shape[-1] == z_quantized.shape[-1]
            z_quantized = decode_mask_rate[:, None, None] * z_quantized
        else:
            # decode_mask rate: [B,]
            if self.regularization_name == "matryoshka":
                z_quantized = self.matryoshka_masking(
                    z_quantized, mask_rate=decode_mask_rate
                )
            else:
                raise NotImplementedError(
                    f"Unsupported reconstruction regularization {self.reconstruction_regularization}."
                )

        # z_quantized.shape: [batch_size, token_dim, 1, num_tokens]
        decoded = self.decoder(z_quantized, key_padding_mask=key_padding_mask)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                "nchw,cd->ndhw",
                decoded.softmax(1),
                self.pixel_quantize.embedding.weight,
            )
            decoded = self.pixel_decoder(quantized_states)
        # decoded.shape: [batch_size, 1024, H, W]
        return decoded

    def encode_tokens(
        self,
        images,
        dino_input=None,
        fixed_mask_rate_val=0.0,
        use_fixed_mask_rate=False,
        vae_results=None,
    ):
        """Encode images to tokens.

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
            print("\033[91mCHECK Not recommended settings: dino_input is None\033[0m")
            dino_input = torch.nn.functional.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Get token features from DINO
        with torch.no_grad():
            token_features = self.feature_extractor(dino_input).last_hidden_state
        if self.use_policy and not use_fixed_mask_rate:
            # Use policy net to estimate the mask rate
            z_quantized, encode_dict = self.encode(
                images, token_features, vae_results=vae_results
            )
        else:
            forward_mask_rate = self.get_mask_rate(images, fixed_mask_rate_val)
            z_quantized, encode_dict = self.encode(
                images, token_features, forward_mask_rate
            )
        full_tokens = encode_dict["min_encoding_indices"].reshape(images.shape[0], -1)
        mask_rate = encode_dict["sampled_mask_rate"].reshape(images.shape[0], -1)
        return full_tokens, mask_rate

    def decode_tokens(self, tokens, decode_mask_rate=0.0):
        """Decode tokens to images.

        Args:
            tokens (Tensor): Input tokens
            decode_mask_rate (float, optional): Decode mask rate. Defaults to 0.0

        Returns:
            Tensor: Decoded images
        """
        if self.quantize_mode == "vq":
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
        elif self.quantize_mode == "vae":
            z_quantized = tokens
            raise ValueError("Unsupported type")
        decode_mask_rate = 1 - seq_len / self.num_latent_tokens
        decoded = self.decode(z_quantized, decode_mask_rate=decode_mask_rate)
        return decoded

    def matryoshka_masking(self, z_quantized, mask_rate):
        """Apply matryoshka masking.

        Args:
            z_quantized (Tensor): Quantized features
            mask_rate (Tensor): Mask rate

        Returns:
            Tensor: Masked features
        """
        # outside function should ensure that mask_rate is meaningful
        # e.g. belong to [0, 1)
        keep_tokens = (
            torch.ceil(z_quantized.shape[-1] * (1 - mask_rate))
            .long()
            .to(z_quantized.device)
        )
        mask = (
            torch.arange(z_quantized.shape[-1], device=z_quantized.device)[None]
            < keep_tokens[:, None]
        )
        return torch.where(mask[:, None, None], z_quantized, 0)

    def forward(
        self,
        x,
        dino_input=None,
        fixed_mask_rate_val=0.0,
        use_fixed_mask_rate=False,
        vae_results=None,
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

        # QY: If dino_input is not provided, use the original image to form the DINO input
        if dino_input is None:
            print("\033[91mCHECK Not recommended settings: dino_input is None\033[0m")
            dino_input = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # 0: Get token features from DINO
        with torch.no_grad():
            token_features = self.feature_extractor(dino_input).last_hidden_state

        # 1: MASKED ENCODING
        if self.use_policy and not use_fixed_mask_rate:
            # Use policy net to estimate the mask rate
            z_quantized, result_dict = self.encode(
                x, token_features, vae_results=vae_results
            )
            result_dict["annealing_factor"] = self.annealing_factor
            forward_mask_rate = result_dict["sampled_mask_rate"]
        else:
            forward_mask_rate = self.get_mask_rate(x, fixed_mask_rate_val)
            z_quantized, result_dict = self.encode(x, token_features, forward_mask_rate)
            result_dict["sampled_mask_rate"] = forward_mask_rate
            result_dict["mask_rate_value"] = forward_mask_rate

        # 2: MASKED DECODING
        decoded = self.decode(z_quantized, decode_mask_rate=forward_mask_rate)

        return decoded, result_dict
