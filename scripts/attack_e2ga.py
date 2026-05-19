"""
ISIC-only E2GA attack script for conditional diffusion-based segmentation.

This script generates adversarial conditional inputs for ISIC images under an
L_infinity perturbation constraint and saves both the adversarial images and
their corresponding diffusion segmentation outputs.

Typical usage:
    python scripts/attack_e2ga.py \
        --data_name ISIC \
        --data_dir ./data/ISIC/Test \
        --model_path emasavedmodel_0.9999_580000.pt \
        --out_dir ./outputs/e2ga_isic \
        --image_size 256 \
        --num_channels 128 \
        --class_cond False \
        --num_res_blocks 2 \
        --num_heads 1 \
        --learn_sigma False \
        --use_scale_shift_norm False \
        --attention_resolutions 16 \
        --diffusion_steps 1000 \
        --noise_schedule linear \
        --rescale_learned_sigmas False \
        --rescale_timesteps False \
        --num_ensemble 1 \
        --batch_size 1 \
        --epsilon 0.031372549 \
        --step_size 0.0039215686 \
        --attack_steps 20
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

sys.path.append(".")

import numpy as np
import torch as th
import torchvision.transforms as transforms
import torchvision.utils as vutils

from guided_diffusion import dist_util, logger
from guided_diffusion.isicloader import ISICDataset3
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible adversarial example generation."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def build_isic_dataset(data_dir: str, image_size: int) -> ISICDataset3:
    """Build the ISIC image-only dataset used as conditional inputs."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return ISICDataset3(data_dir, transform)


def load_checkpoint(model: th.nn.Module, checkpoint_path: str) -> None:
    """Load a pretrained model checkpoint.

    This restores the original checkpoint-loading logic used in the experimental
    script. If keys contain the DistributedDataParallel prefix ``module.``, the
    prefix is removed before loading.
    """
    if not checkpoint_path:
        raise ValueError("Please provide --model_path for the pretrained checkpoint.")

    state_dict = dist_util.load_state_dict(checkpoint_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove `module.` prefix if the checkpoint was saved from DDP
        if "module." in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)


def extract_isic_id(path_item) -> str:
    """Extract an ISIC sample identifier from the dataset-provided filename."""
    path_str = str(path_item)
    stem = Path(path_str).stem
    return stem.split("_")[-1]


def e2ga_update(
        diffusion,
        model: th.nn.Module,
        adv_input: th.Tensor,
        clean_input: th.Tensor,
        sampling_noise: th.Tensor,
        args: argparse.Namespace,
) -> Tuple[th.Tensor, th.Tensor]:
    """Run the E2GA PGD loop for one ISIC batch.

    Args:
        diffusion: Diffusion sampler containing ``p_sample_loop_known``.
        model: Conditional diffusion segmentation model.
        adv_input: Current adversarial conditional input.
        clean_input: Original clean conditional input.
        sampling_noise: Fixed auxiliary noise channel for the reverse chain.
        args: Command-line arguments.

    Returns:
        A tuple ``(adv_input, sample)`` containing the final adversarial input
        and the last sampled segmentation output.
    """
    epsilon = float(args.epsilon)
    step_size = float(args.step_size)

    sample = None
    sample_shape = (
        adv_input.shape[0],
        3,
        args.image_size,
        args.image_size,
    )

    for _ in range(args.attack_steps):
        conditional_input = th.cat((adv_input, sampling_noise), dim=1)

        sample, _, _, _, _, grad = diffusion.p_sample_loop_known(
            model,
            sample_shape,
            conditional_input,
            adv_input,
            step=args.diffusion_steps,
            clip_denoised=args.clip_denoised,
            model_kwargs={},
        )

        if grad is None:
            raise RuntimeError(
                "The diffusion sampler did not return a gradient. "
                "Please check that guided_diffusion/gaussian_diffusion.py contains "
                "the E2GA gradient-returning implementation of p_sample_loop_known."
            )

        grad = grad.to(adv_input.device)
        adv_input = adv_input + step_size * th.sign(grad)
        adv_input = th.clamp(adv_input, clean_input - epsilon, clean_input + epsilon)
        adv_input = th.clamp(adv_input, 0.0, 1.0).detach().requires_grad_(True)

    return adv_input, sample


def save_adversarial_results(
        diffusion,
        model: th.nn.Module,
        adv_input: th.Tensor,
        sampling_noise: th.Tensor,
        sample_id: str,
        args: argparse.Namespace,
) -> th.Tensor:
    """Save the adversarial ISIC image and its generated segmentation output."""
    adv_dir = Path(args.out_dir) / "adv_images"
    output_dir = Path(args.out_dir) / "adv_output"
    adv_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        adv_input.detach(),
        fp=str(adv_dir / f"ISIC_{sample_id}_adversarial.{args.output_ext}"),
        nrow=1,
        padding=0,
    )

    conditional_input = th.cat((adv_input, sampling_noise), dim=1)
    sample_shape = (
        adv_input.shape[0],
        3,
        args.image_size,
        args.image_size,
    )

    sample, _, _, _, _, _ = diffusion.p_sample_loop_known(
        model,
        sample_shape,
        conditional_input,
        adv_input,
        step=args.diffusion_steps,
        clip_denoised=args.clip_denoised,
        model_kwargs={},
    )

    vutils.save_image(
        sample[:, -1:, ...].detach(),
        fp=str(output_dir / f"ISIC_{sample_id}_output_ens.{args.output_ext}"),
        nrow=1,
        padding=0,
    )

    return sample


def main() -> None:
    args = create_argparser().parse_args()

    if args.use_ddim:
        raise NotImplementedError(
            "This ISIC E2GA implementation requires p_sample_loop_known, because "
            "the current DDIM sampler does not return the pathwise input gradient."
        )

    set_seed(args.seed)
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    # ISIC images are RGB conditional inputs, and the diffusion segmentation
    # model receives RGB + one noisy segmentation channel.
    args.in_ch = 4

    dataset = build_isic_dataset(args.data_dir, args.image_size)
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=th.cuda.is_available(),
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    load_checkpoint(model, args.model_path)

    device = dist_util.dev()
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    output_means = []
    for clean_input, path in dataloader:
        clean_input = clean_input.to(device)
        adv_input = clean_input.clone().detach().requires_grad_(True)
        sampling_noise = th.randn_like(clean_input[:, :1, ...], device=device)

        sample_id = extract_isic_id(path[0])
        logger.log(f"attacking ISIC sample {sample_id}...")

        start = th.cuda.Event(enable_timing=True) if th.cuda.is_available() else None
        end = th.cuda.Event(enable_timing=True) if th.cuda.is_available() else None
        if start is not None:
            start.record()

        adv_input, _ = e2ga_update(
            diffusion=diffusion,
            model=model,
            adv_input=adv_input,
            clean_input=clean_input,
            sampling_noise=sampling_noise,
            args=args,
        )

        sample = save_adversarial_results(
            diffusion=diffusion,
            model=model,
            adv_input=adv_input,
            sampling_noise=sampling_noise,
            sample_id=sample_id,
            args=args,
        )

        if end is not None:
            end.record()
            th.cuda.synchronize()
            logger.log(f"time for ISIC sample {sample_id}: {start.elapsed_time(end):.2f} ms")

        output_mean = sample[:, -1:, ...].mean().item()
        output_means.append(output_mean)
        logger.log(f"output mean for ISIC sample {sample_id}: {output_mean:.6f}")

    if output_means:
        logger.log(f"mean output value over all ISIC samples: {float(np.mean(output_means)):.6f}")


def create_argparser() -> argparse.ArgumentParser:
    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            # Dataset / path settings
            data_name="ISIC",
            data_dir="./data/ISIC/Test",
            model_path="emasavedmodel_0.9999_580000.pt",
            out_dir="./outputs/e2ga_isic",

            # ISIC MedSegDiff checkpoint configuration.
            # These settings must match the original segmentation.py command
            # used with emasavedmodel_0.9999_580000.pt.
            image_size=256,
            num_channels=128,
            class_cond=False,
            num_res_blocks=2,
            num_heads=1,
            learn_sigma=False,
            use_scale_shift_norm=False,
            attention_resolutions="16",
            diffusion_steps=1000,
            noise_schedule="linear",
            rescale_learned_sigmas=False,
            rescale_timesteps=False,
            num_ensemble=1,

            # Sampling / attack settings
            clip_denoised=True,
            num_samples=1,
            batch_size=1,
            use_ddim=False,
            gpu_dev="0",
            multi_gpu=None,
            debug=False,
            seed=10,
            epsilon=8 / 255,
            step_size=1 / 255,
            attack_steps=20,
            output_ext="png",
            num_workers=0,
        )
    )

    parser = argparse.ArgumentParser(
        description="Run the ISIC-only E2GA attack on a conditional diffusion segmentation model."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()