"""This files contains training loss implementation.

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

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor,
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(
        torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2)
    )
    lecam_loss += torch.mean(
        torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2)
    )
    return lecam_loss


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class FlowMatchingLoss_Stage1(torch.nn.Module):
    """This loss is reproduced from FlexTok: https://www.arxiv.org/pdf/2502.13967
    
        The loss including the flow matching loss, quantization loss and the repa loss.
    """
    def __init__(self, config):
        super().__init__()
        loss_config = config.losses
        self.config = config
        self.quantizer_weight = loss_config.quantizer_weight
        self.repa_weight = loss_config.get("repa_weight", 1.0)
        self.rate_weight = config.model.reconstruction_regularization.policy.rate_weight
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(
        self,
        place_holder: torch.Tensor,
        predicted_flow: torch.Tensor,
        extra_input_dict: Mapping[Text, torch.Tensor],
        mode: str = "with_ground_truth",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Under flow matching training, place_holder (corresponding to the original proxy_code) is not used
        return self._forward_generator(
            place_holder, predicted_flow, extra_input_dict, mode=mode
        )

    def _forward_generator(
        self,
        place_holder: torch.Tensor,
        predicted_flow: torch.Tensor,
        extra_input_dict: Mapping[Text, torch.Tensor],
        mode: str = "with_ground_truth",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        if mode == "with_policy":
            # flow_target.shape, predicted_flow.shape: [B, 16, 32, 32]
            flow_matching_loss = self.loss_fn(
                extra_input_dict["flow_target"], predicted_flow
            ).mean(
                dim=(1, 2, 3)
            )  # [B,]
            rate_loss = 1 - extra_input_dict["mask_rate_value"]  # [B,]
            critic_loss = flow_matching_loss + self.rate_weight * rate_loss

            reward = critic_loss.detach()
            actor_loss = reward * extra_input_dict["logprob_mask"]
            critic_loss, actor_loss = critic_loss.mean(), actor_loss.mean()

            # REPA loss: https://arxiv.org/abs/2410.06940
            dino_feature = F.normalize(extra_input_dict["dino_feature"], dim=-1)
            repa_feature = F.normalize(extra_input_dict["repa_feature"], dim=-1)
            repa_loss = torch.mean(-(dino_feature * repa_feature).sum(dim=-1))

            total_loss = (
                critic_loss
                + extra_input_dict["annealing_factor"] * actor_loss
                + self.quantizer_weight * extra_input_dict["quantizer_loss"]
                + self.repa_weight * repa_loss
            )

            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=flow_matching_loss.mean().detach(),
                reconstruction_loss_unreduced=flow_matching_loss.detach(),
                rate_loss=rate_loss.mean().detach(),
                rate_loss_unreduced=rate_loss.detach(),
                rate_std=rate_loss.std().detach(),  # sample wise variance
                actor_loss=actor_loss.detach(),
                critic_loss=critic_loss.detach(),
                quantizer_loss=(
                    self.quantizer_weight * extra_input_dict["quantizer_loss"]
                ).detach(),
                commitment_loss=extra_input_dict["commitment_loss"].detach(),
                codebook_loss=extra_input_dict["codebook_loss"].detach(),
                repa_loss=repa_loss.detach(),
            )

            return total_loss, loss_dict

        if mode == "with_ground_truth":
            flow_matching_loss = self.loss_fn(
                extra_input_dict["flow_target"], predicted_flow
            ).mean(
                dim=(1, 2, 3)
            )  # [B,]
        else:
            raise ValueError(f"Unsupported loss mode {mode}")

        total_loss = (
            flow_matching_loss.mean()
            + self.quantizer_weight * extra_input_dict["quantizer_loss"]
        )

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=flow_matching_loss.mean().detach(),
            reconstruction_loss_unreduced=flow_matching_loss.detach(),
            quantizer_loss=(
                self.quantizer_weight * extra_input_dict["quantizer_loss"]
            ).detach(),
            commitment_loss=extra_input_dict["commitment_loss"].detach(),
            codebook_loss=extra_input_dict["codebook_loss"].detach(),
        )

        return total_loss, loss_dict


class ReconstructionLoss_Stage1(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        loss_config = config.losses
        self.config = config
        self.quantizer_weight = loss_config.quantizer_weight
        self.rate_weight = config.model.reconstruction_regularization.policy.rate_weight
        self.target_codebook_size = 1024

        try:
            self.use_gumbel_softmax = (
                config.model.reconstruction_regularization.use_gumbel_softmax
            )
        except:
            self.use_gumbel_softmax = False

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        target_codes: torch.Tensor,
        reconstructions: torch.Tensor,
        quantizer_loss: torch.Tensor,
        mode: str = "with_ground_truth",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(
            target_codes, reconstructions, quantizer_loss, mode=mode
        )

    def _forward_generator(
        self,
        target_codes: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_input_dict: Mapping[Text, torch.Tensor],
        mode: str = "with_ground_truth",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        reconstructions = reconstructions.contiguous()
        if mode == "with_policy":
            batch_size = reconstructions.shape[0]

            reconstruction_loss = self.loss_fn(
                reconstructions.view(batch_size, self.target_codebook_size, -1),
                target_codes.view(batch_size, -1),
            ).mean(
                dim=1
            )  # [B,]
            rate_loss = 1 - extra_input_dict["mask_rate_value"]  # [B,]

            critic_loss = reconstruction_loss + self.rate_weight * rate_loss

            if (
                self.config.model.reconstruction_regularization.use_gumbel_softmax
            ):  # The reinforce framework
                actor_loss = torch.zeros_like(critic_loss)

            else:
                reward = critic_loss.detach()
                actor_loss = reward * extra_input_dict["logprob_mask"]

            critic_loss, actor_loss = critic_loss.mean(), actor_loss.mean()

            total_loss = (
                critic_loss
                + extra_input_dict["annealing_factor"] * actor_loss
                + self.quantizer_weight * extra_input_dict["quantizer_loss"]
            )

            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.mean().detach(),
                reconstruction_loss_unreduced=reconstruction_loss.detach(),
                rate_loss=rate_loss.mean().detach(),
                rate_loss_unreduced=rate_loss.detach(),
                rate_std=rate_loss.std().detach(),  # sample wise variance
                actor_loss=actor_loss.detach(),
                critic_loss=critic_loss.detach(),
                quantizer_loss=(
                    self.quantizer_weight * extra_input_dict["quantizer_loss"]
                ).detach(),
                commitment_loss=extra_input_dict["commitment_loss"].detach(),
                codebook_loss=extra_input_dict["codebook_loss"].detach(),
            )

            return total_loss, loss_dict

        batch_size = reconstructions.shape[0]
        if mode == "with_ground_truth":
            # DEBUG: in this case, target_codes is indices of codebook
            # reconstructions.shape: [batch_size, codebook_size, H, W]
            # target_codes.shape: [batch_size, H * W]
            reconstruction_loss = self.loss_fn(
                reconstructions.view(batch_size, self.target_codebook_size, -1),
                target_codes.view(batch_size, -1),
            ).mean(
                dim=-1
            )  # [B,]

        else:
            raise ValueError(f"Unsupported loss mode {mode}")

        total_loss = (
            reconstruction_loss.mean()
            + self.quantizer_weight * extra_input_dict["quantizer_loss"]
        )

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.mean().detach(),
            reconstruction_loss_unreduced=reconstruction_loss.detach(),
            quantizer_loss=(
                self.quantizer_weight * extra_input_dict["quantizer_loss"]
            ).detach(),
            commitment_loss=extra_input_dict["commitment_loss"].detach(),
            codebook_loss=extra_input_dict["codebook_loss"].detach(),
        )

        return total_loss, loss_dict


class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(self, config):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start
        self.rate_weight = config.model.reconstruction_regularization.policy.rate_weight

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

    @autocast(enabled=False)
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(
                inputs, reconstructions, extra_result_dict, global_step
            )
        elif mode == "discriminator":
            return self._forward_discriminator(
                inputs, reconstructions, extra_result_dict, global_step
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            # reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
            reconstruction_loss = F.l1_loss(
                inputs, reconstructions, reduction="none"
            )  # For unreduced logout
        elif self.reconstruction_loss == "l2":
            # reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
            reconstruction_loss = F.mse_loss(
                inputs, reconstructions, reduction="none"
            )  # For unreduced logout
        else:
            raise ValueError(
                f"Unsuppored reconstruction_loss {self.reconstruction_loss}"
            )
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Consider rate loss
        rate_loss_unreduced = 1 - extra_result_dict["mask_rate_value"]  # (B,)

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            reconstruction_loss.mean()
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.mean().detach(),
            reconstruction_loss_unreduced=reconstruction_loss.mean(
                dim=(1, 2, 3)
            ).detach(),  # keep batch info
            rate_loss=rate_loss_unreduced.mean().detach(),
            rate_loss_unreduced=rate_loss_unreduced.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(
                d_weight * discriminator_factor * generator_loss
            ).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = (
                compute_lecam_loss(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean,
                )
                * self.lecam_regularization_weight
            )

            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class MLMLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing, reduction="none"
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, weights=None
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (
            1.0 - weights
        ) * self.loss_weight_unmasked_token + weights  # set 0 to self.loss_weight_unasked_token
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(
            dim=1
        ) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}


class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_weight_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Target of last token is meaningless, match next token
        shift_logits = (
            logits[..., :-1, :].permute(0, 2, 1).contiguous()
        )  # (B, [cond]+S+[eos], codebook_size) -> (B, codebook_size, [cond]+S)
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(
            shift_logits.shape[0], self.target_vocab_size, -1
        )
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(
            shift_logits, shift_labels
        )  # (B, codebook_size, [cond]+S)
        loss = torch.mean(loss, dim=-2)  # (B, [cond]+S)
        loss = loss * loss_weight_mask  # ignore the irrelevant padding tokens
        loss = loss.sum() / loss_weight_mask.sum()  # weighted average
        correct_tokens = (
            torch.argmax(shift_logits, dim=1) == shift_labels
        ) * loss_weight_mask
        correct_tokens = correct_tokens.sum() / loss_weight_mask.sum()
        return loss, {"loss": loss, "correct_tokens": correct_tokens}
