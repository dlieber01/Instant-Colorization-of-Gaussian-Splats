"""
Color reconstruction for a fixed 3D Gaussian geometry.

This script assumes that the Gaussian positions, scales, rotations, and opacities
have already been optimized. We then re-estimate only the appearance parameters
from image evidence.

The implementation exposes two color-reconstruction modes:

1. ``adam``:
   Direct gradient-based optimization of the SH coefficients by minimizing image
   reconstruction error over the training views.

2. ``instant``:
   A solver-based initialization that estimates SH coefficients from accumulated
   per-view visibility and color-gradient statistics, followed by a small number
   of refinement steps.
"""
import os
import random
import torch
import time
import torch.nn.functional as F
import json
from tqdm import tqdm
from pytorch_msssim import ssim as msssim_ssim
from torch.amp import autocast
from pathlib import Path

from scene import Scene
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from argparse import ArgumentParser
from utils.sh_utils import C0
import cv2

# Prevent potential conflicts with Intel MKL
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Per-order Tikhonov regularization weights for the SH coefficients.
# Lower-order terms are regularized less aggressively because they carry the
# coarse appearance, whereas higher-order terms mainly model view-dependent
# residual detail and are therefore more noise-sensitive.
LAMBDAS = [1e-5, 1e-2, 1e-2, 1e-2]
LAMBDAS = torch.tensor(
    [
        LAMBDAS[0],
        LAMBDAS[1],LAMBDAS[1],LAMBDAS[1],
        LAMBDAS[2],LAMBDAS[2],LAMBDAS[2],LAMBDAS[2],LAMBDAS[2],
        LAMBDAS[3],LAMBDAS[3],LAMBDAS[3],LAMBDAS[3],LAMBDAS[3],LAMBDAS[3],LAMBDAS[3],
    ]
).unsqueeze(0)

# Resolution downsampling used when loading images.
RESOLUTION = 2  # -1 means original resolution, 2 means half resolution, 4 means quarter resolution etc.
n_per_modulo = 1
modulo = 1
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_PATH = PROJECT_ROOT / "data"


def plot_image(image, gt=None):
    """Visualize the current rendering and, optionally, the ground-truth image."""
    if gt is not None:
        cv2.imshow(
            "ground truth", gt.detach().cpu().permute(1, 2, 0).numpy()[:, :, (2, 1, 0)]
        )
    cv2.imshow("render", image.detach().cpu().permute(1, 2, 0).numpy()[:, :, (2, 1, 0)])
    key = cv2.waitKey(1)

    if key == ord("q"):
        exit()


def filter_train_cameras(train_cameras, modulo=modulo, n_per_modulo=n_per_modulo):
    """
    Subsample the training cameras in a deterministic modulo pattern.

    This is useful for experiments where only a structured subset of the
    available views should contribute to the color reconstruction.
    """
    """
    :train_images: list of train cameras from scene
    :modulo: e.g. return every 3. or every 4. camera
    :n_per_modulo: e.g. if 2: return camera 0,1,5,6,10,11,15,16, ...
    :return: list of filtered train_images
    """
    return [tc for i, tc in enumerate(train_cameras) if i % modulo < n_per_modulo]


def tensor_to_jsonable(x):
    """Convert tensors into plain Python containers for JSON serialization."""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def resolve_device() -> torch.device:
    """Select CUDA when available; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        return torch.device("cuda")

    print("CUDA is not available. Running on CPU.")
    return torch.device("cpu")


LIGHTING_SPLITS = {
    "LLFF_dataset/horns_seg": [63],
    "LLFF_dataset/trex_seg": [56],
    "LLFF_dataset/trex_seg2": [56],
    "LLFF_dataset/trex_seg3": [56],
    "LLFF_dataset/fortress_seg": [43],
    "table_garden": [24],
    "table_garden_seg": [24],
    "counter": [28],
    "scene5_all": [49, 63, 74, 77, 89],
    "scene6_all": [79, 80, 75, 78, 84],
    "scene7_all": [78, 78, 81, 84, 76],
    "scene5_all_dino": [49, 63, 74, 77, 89],
    "scene6_all_dino": [79, 80, 75, 78, 84],
    "scene7_all_dino": [78, 78, 81, 84, 76],
    "scene5_all_segmented_multi_items": [49, 63, 74, 77, 89],
    "scene6_all_segmented_multi_items": [79, 80, 75, 78, 84],
    "scene7_all_segmented_multi_items": [78, 78, 81, 84, 76],
    "scene5_all_masked_multi_items": [49, 63, 74, 77, 89],
    "scene6_all_masked_multi_items": [79, 80, 75, 78, 84],
    "scene7_all_masked_multi_items": [78, 78, 81, 84, 76],
}


def setup_plot_windows(plot_images: bool) -> None:
    """Create OpenCV windows only when interactive visualization is enabled."""
    if not plot_images:
        return
    cv2.namedWindow("ground truth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("render", cv2.WINDOW_NORMAL)


def build_model_and_pipeline_params(scene_path, iteration, source_path_arg, device):
    """Prepare the project-specific parameter containers used by Scene."""
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    scene_folder = Path(scene_path).name
    source_name = scene_folder.split("_")[0]

    model_params.model_path = scene_path
    model_params.load_iteration = iteration
    model_params.source_path = (
        source_path_arg if source_path_arg is not None else str(BASE_DATA_PATH / source_name)
    )
    model_params.images = None
    model_params.depths = ""
    model_params.resolution = RESOLUTION
    model_params.eval = True
    model_params.train_test_exp = False
    model_params.white_background = False
    model_params.data_device = device
    return model_params, pipeline_params


def compute_lighting_split_indices(source_path, lighting_index):
    """Return train/test slice bounds for datasets containing multiple light setups."""
    source_key = Path(source_path).name
    if source_key not in LIGHTING_SPLITS:
        return None, None

    lighting_n_images = LIGHTING_SPLITS[source_key]
    if lighting_index < 0 or lighting_index >= len(lighting_n_images):
        raise IndexError(
            f"lighting_index={lighting_index} is out of range for source '{source_key}' "
            f"with {len(lighting_n_images)} lighting blocks."
        )

    lighting_indices = []
    start_index = 0
    for n in lighting_n_images:
        lighting_indices.append(list(range(start_index, start_index + n)))
        start_index += n

    end_train_index = 0
    end_test_index = 0
    for i in range(lighting_index + 1):
        start_train_index = end_train_index
        start_test_index = end_test_index
        end_train_index += len([x for x in lighting_indices[i] if x % 8 != 0])
        end_test_index += len([x for x in lighting_indices[i] if x % 8 == 0])

    return [start_train_index, end_train_index], [start_test_index, end_test_index]


def load_scene_bundle(scene_path, iteration, source_path_arg, lighting_index, device):
    """Load the fixed Gaussian geometry together with its scene and camera setup."""
    model_params, pipeline_params = build_model_and_pipeline_params(
        scene_path, iteration, source_path_arg, device
    )
    safe_state(False)

    gaussians = GaussianModel(model_params.sh_degree)
    train_indices, test_indices = compute_lighting_split_indices(
        model_params.source_path, lighting_index
    )
    scene = Scene(
        model_params,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    bg_color = torch.zeros(3, device=device)
    cam_list = filter_train_cameras(scene.getTrainCameras())
    if not cam_list:
        raise RuntimeError(
            "No training cameras found after filtering. "
            "Check train/test split, lighting_index, and filter_train_cameras()."
        )

    print(f"n_train: {len(cam_list)} / n_test: {len(scene.getTestCameras())}")
    return {
        "model_params": model_params,
        "pipeline_params": pipeline_params,
        "gaussians": gaussians,
        "scene": scene,
        "bg_color": bg_color,
        "cam_list": cam_list,
        "cam_0": cam_list[0],
    }


def create_colorization_log(model_params, lighting_index, color_opt):
    return {
        "source_path": model_params.source_path,
        "lighting_index": int(lighting_index),
        "color_opt": str(color_opt),
        "LAMBDAS": tensor_to_jsonable(LAMBDAS),
        "train_stats": [],
        "test_stats": [],
    }


def append_metric_stage(colorization_log, stage, train_metrics, test_metrics):
    colorization_log["train_stats"].append(
        {"stage": stage, **{k: float(v) for k, v in train_metrics.items()}}
    )
    colorization_log["test_stats"].append(
        {"stage": stage, **{k: float(v) for k, v in test_metrics.items()}}
    )


def evaluate_and_log_metrics(
    gaussians,
    stage,
    scene,
    pipeline_params,
    bg_color,
    colorization_log=None,
    headline=None,
):
    """Evaluate train/test metrics, optionally append them to the run log."""
    train_metrics = compute_losses(
        gaussians,
        filter_train_cameras(scene.getTrainCameras()),
        pipeline_params,
        bg_color,
    )
    test_metrics = compute_losses(
        gaussians,
        scene.getTestCameras(),
        pipeline_params,
        bg_color,
    )
    if colorization_log is not None:
        append_metric_stage(colorization_log, stage, train_metrics, test_metrics)
    if headline is None:
        headline = f"Metrics ({stage})"
    print(
        f"{headline} — "
        f"Train: L1={train_metrics['l1']:.6f} L2={train_metrics['l2']:.6f} "
        f"SSIM={train_metrics['ssim']:.6f} PSNR={train_metrics['psnr']:.2f} | "
        f"Test:  L1={test_metrics['l1']:.6f} L2={test_metrics['l2']:.6f} "
        f"SSIM={test_metrics['ssim']:.6f} PSNR={test_metrics['psnr']:.2f}"
    )
    return train_metrics, test_metrics


def reset_gaussian_colors(gaussians, device, sh_degree_n=None):
    """Keep the geometry and replace all color coefficients by zeros."""
    gaussians._features_dc = torch.zeros_like(
        gaussians._features_dc, device=device, requires_grad=True
    )
    if sh_degree_n is None:
        gaussians._features_rest = torch.zeros_like(
            gaussians._features_rest, device=device, requires_grad=True
        )
    else:
        gaussians._features_rest = torch.zeros_like(
            gaussians._features_rest[:, : (sh_degree_n - 1)],
            device=device,
            requires_grad=True,
        )


def maybe_clear_cuda_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_adam_color_optimization(
    gaussians,
    cam_list,
    cam_0,
    pipeline_params,
    bg_color,
    device,
    plot_images,
    loss_history,
):
    """Optimize SH coefficients directly with image-space gradients."""
    reset_gaussian_colors(gaussians, device)
    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians._features_dc], "lr": 0.0025},
            {"params": [gaussians._features_rest], "lr": (0.0025 / 20)},
        ]
    )

    start_time_adam = time.time()
    for _ in tqdm(range(100), desc="Adam Color Optimization"):
        shuffled_cams = random.sample(cam_list, len(cam_list))
        for cam in shuffled_cams:
            optimizer.zero_grad()
            gt_image = cam.original_image.to(device)
            with autocast(device_type=device.type):
                render_pkg = render(cam, gaussians, pipeline_params, bg_color)
                img = render_pkg["render"]
                if plot_images and cam == cam_0:
                    plot_image(img, gt_image)
                loss = F.mse_loss(img, gt_image)
            loss.backward()
            optimizer.step()

            elapsed = time.time() - start_time_adam
            loss_history.append((elapsed, loss.item()))

            del render_pkg, img, gt_image, loss
            maybe_clear_cuda_cache(device)

    end_time_adam = time.time()
    print(f"Adam optimization completed in {end_time_adam - start_time_adam:.1f} s.")
    return gaussians.clone()


def collect_solver_statistics(
    gaussians,
    cam_list,
    cam_0,
    pipeline_params,
    bg_color,
    device,
    plot_images,
    sh_degree_n,
    n_channels,
):
    """Accumulate the sufficient statistics used by the instant solver."""
    n_gaussians = gaussians.get_xyz.shape[0]
    n_cams = len(cam_list)
    color_matrix = torch.zeros((n_gaussians, n_channels, n_cams), device=device)
    visibility_matrix = torch.zeros((n_gaussians, 1, n_cams), device=device)
    sh_matrix = torch.zeros((n_gaussians, sh_degree_n, n_cams), device=device)
    sh_matrix[:, 0, :] = C0
    gt_images = []

    for i, cam in tqdm(enumerate(cam_list), total=n_cams, desc="Solver Coloring"):
        gt_images.append(cam.original_image.to(device))
        render_pkg = render(cam, gaussians, pipeline_params, bg_color)
        img = render_pkg["render"]

        if plot_images and cam == cam_0:
            plot_image(img, gt_images[i])

        sum_img = torch.sum(img)
        gaussians.zero_grad()
        sum_img.backward(retain_graph=True)
        vis = gaussians._features_dc.grad.data.mean(2, keepdim=False).detach() / C0
        visibility_matrix[:, :, i] = vis

        if sh_degree_n > 1:
            sh_matrix[:, 1:, i] = (
                gaussians._features_rest.grad.data.mean(2, keepdim=False).detach()
                / vis.clamp(min=1e-6)
            )

        sum_delta_img = torch.sum(gt_images[i] * img)
        gaussians.zero_grad()
        sum_delta_img.backward()
        rgb = (
            gaussians._features_dc.grad.data.detach()[:, 0, :n_channels]
            / C0
            / vis.clamp(min=1e-6)
        )
        color_matrix[:, :, i] = rgb - 0.5

        del render_pkg, img, rgb, vis, sum_img, sum_delta_img
        maybe_clear_cuda_cache(device)

    return {
        "color_matrix": color_matrix,
        "visibility_matrix": visibility_matrix,
        "sh_matrix": sh_matrix,
        "gt_images": gt_images,
    }


def solve_color_system(color_matrix, visibility_matrix, sh_matrix, sh_degree_n, device):
    """Solve the regularized normal equations for all Gaussians in parallel."""
    sh_visibility_matrix = sh_matrix * visibility_matrix
    a = torch.einsum("ijl,inl->ijn", sh_visibility_matrix, sh_matrix)
    b = torch.einsum(
        "ijkl,ijkl->ijk",
        sh_visibility_matrix.unsqueeze(2),
        color_matrix.unsqueeze(1),
    )
    visibility_sum = visibility_matrix.sum(dim=2).clamp(min=1e-30)
    regularizer = visibility_sum * LAMBDAS[:, :sh_degree_n].to(device)
    a_regularized = a + torch.diag_embed(regularizer)
    x = torch.linalg.solve(a_regularized, b)
    return x, sh_visibility_matrix, regularizer, a_regularized


def apply_solver_solution(gaussians, solution, sh_degree_n):
    """Write the solved SH coefficients back into a GaussianModel clone."""
    colorized_gaussians = gaussians.clone()
    colorized_gaussians._features_dc.data[:] = solution[:, :1, :]
    if sh_degree_n > 1:
        colorized_gaussians._features_rest.data[:] = solution[:, 1:, :]
    return colorized_gaussians


def run_instant_color_optimization(
    gaussians,
    scene,
    cam_list,
    cam_0,
    pipeline_params,
    bg_color,
    device,
    plot_images,
    save_loss,
    colorization_log,
    refinement_steps,
    sh_degree_n,
    n_channels,
):
    """Run solver initialization followed by optional refinement iterations."""
    reset_gaussian_colors(gaussians, device, sh_degree_n=sh_degree_n)
    start_time_solver = time.time()
    stats = collect_solver_statistics(
        gaussians,
        cam_list,
        cam_0,
        pipeline_params,
        bg_color,
        device,
        plot_images,
        sh_degree_n,
        n_channels,
    )
    x, sh_visibility_matrix, regularizer, a_regularized = solve_color_system(
        stats["color_matrix"],
        stats["visibility_matrix"],
        stats["sh_matrix"],
        sh_degree_n,
        device,
    )
    print(
        f"Instant solver computation completed in {time.time() - start_time_solver:.1f} s."
    )

    colorized_gaussians = apply_solver_solution(gaussians, x, sh_degree_n)

    if save_loss:
        print("Compute metrics after solver initialization...")
        evaluate_and_log_metrics(
            colorized_gaussians,
            "solver_init",
            scene,
            pipeline_params,
            bg_color,
            colorization_log=colorization_log,
            headline="Metrics (Solver-Init)",
        )

    for refinement_step in tqdm(
        range(refinement_steps), desc="Refinement Steps", total=refinement_steps
    ):
        step_start_time = time.time()
        for i, cam in enumerate(cam_list):
            render_pkg = render(cam, colorized_gaussians, pipeline_params, bg_color)
            img = render_pkg["render"]

            if plot_images and cam == cam_0:
                plot_image(img, stats["gt_images"][i])

            sum_delta_img = torch.sum((stats["gt_images"][i] - img.detach()) * img)
            colorized_gaussians.zero_grad()
            sum_delta_img.backward()

            rgb = (
                colorized_gaussians._features_dc.grad.data.detach()[:, 0, :]
                / C0
                / stats["visibility_matrix"][:, :, i].clamp(min=1e-6)
            )
            stats["color_matrix"][:, :, i] = rgb.detach()

            del render_pkg, img, rgb, sum_delta_img
            maybe_clear_cuda_cache(device)

        b = torch.einsum(
            "ijkl,ijkl->ijk",
            sh_visibility_matrix.unsqueeze(2),
            stats["color_matrix"].unsqueeze(1),
        )
        if sh_degree_n > 1:
            c = torch.cat(
                [
                    colorized_gaussians._features_dc.detach(),
                    colorized_gaussians._features_rest.detach(),
                ],
                1,
            )
        else:
            c = colorized_gaussians._features_dc.detach()
        b = b - regularizer.unsqueeze(2) * c
        x = torch.linalg.solve(a_regularized, b)

        colorized_gaussians._features_dc.data[:] = (
            colorized_gaussians._features_dc.detach() + x[:, :1, :]
        )
        if sh_degree_n > 1:
            colorized_gaussians._features_rest.data[:] = (
                colorized_gaussians._features_rest.detach() + x[:, 1:, :]
            )

        step_time = time.time() - step_start_time
        print(
            f"Refinement Step {refinement_step + 1}/{refinement_steps} completed in {step_time:.1f}s"
        )

        if save_loss:
            stage = f"refine_{refinement_step + 1}"
            print(f"Compute metrics after {stage}...")
            evaluate_and_log_metrics(
                colorized_gaussians,
                stage,
                scene,
                pipeline_params,
                bg_color,
                colorization_log=colorization_log,
            )

    print(f"Total solver time: {time.time() - start_time_solver:.1f} s")

    if save_loss:
        print("Compute final metrics")
        evaluate_and_log_metrics(
            colorized_gaussians,
            "final",
            scene,
            pipeline_params,
            bg_color,
            colorization_log=colorization_log,
            headline="Final Metrics",
        )

    final_gaussians = colorized_gaussians.clone()
    final_gaussians._features_dc = colorized_gaussians._features_dc
    if sh_degree_n > 1:
        final_gaussians._features_rest = colorized_gaussians._features_rest
    return final_gaussians


def save_colorization_outputs(
    scene_path,
    iteration,
    color_opt,
    lighting_index,
    save_loss,
    loss_history,
    colorization_log,
    final_gaussians,
    output_path,
):
    """Persist logs and the recolored Gaussian point cloud."""
    if save_loss:
        loss_path = os.path.join(scene_path, f"loss_history_{color_opt}.pt")
        torch.save(loss_history, loss_path)
        print(f"Loss history saved under: {loss_path}")

        log_path = os.path.join(
            scene_path, f"metrics_{color_opt}_lighting_{lighting_index}.json"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(colorization_log, f, indent=2)
        print(f"Metrics log saved under: {log_path}")

    if output_path is not None:
        if not output_path.lower().endswith(".ply"):
            raise ValueError(
                f"output_path must be a .ply file, got: {output_path}"
            )
        out_path = output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_dir = os.path.join(scene_path, "point_cloud", f"iteration_colored_{iteration}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"point_cloud_{color_opt}.ply")
    final_gaussians.save_ply(out_path)
    print(f"Saved colorized point cloud to: {out_path}")


def colorize(
    scene_path,
    iteration,
    color_opt,
    source_path_arg,
    refinement_steps,
    sh_degree,
    SAVE_LOSS,
    PLOT_IMAGES,
    lighting_index,
    output_path,
):
    """Top-level orchestration for color reconstruction on a fixed geometry."""
    device = resolve_device()
    setup_plot_windows(PLOT_IMAGES)

    bundle = load_scene_bundle(
        scene_path, iteration, source_path_arg, lighting_index, device
    )
    gaussians = bundle["gaussians"]
    scene = bundle["scene"]
    pipeline_params = bundle["pipeline_params"]
    bg_color = bundle["bg_color"]
    cam_list = bundle["cam_list"]
    cam_0 = bundle["cam_0"]
    model_params = bundle["model_params"]

    sh_degree_n = (sh_degree + 1) ** 2
    gaussians.active_sh_degree = sh_degree
    n_channels = gaussians._features_dc.shape[2]

    loss_history = []
    colorization_log = create_colorization_log(
        model_params, lighting_index, color_opt
    )

    print(f"Starting {color_opt} color optimization with {refinement_steps} iterations…")
    if color_opt == "adam":
        final_gaussians = run_adam_color_optimization(
            gaussians,
            cam_list,
            cam_0,
            pipeline_params,
            bg_color,
            device,
            PLOT_IMAGES,
            loss_history,
        )
        if SAVE_LOSS:
            evaluate_and_log_metrics(
                final_gaussians,
                "final",
                scene,
                pipeline_params,
                bg_color,
                colorization_log=colorization_log,
                headline="Final metrics (Adam)",
            )
    elif color_opt == "instant":
        final_gaussians = run_instant_color_optimization(
            gaussians,
            scene,
            cam_list,
            cam_0,
            pipeline_params,
            bg_color,
            device,
            PLOT_IMAGES,
            SAVE_LOSS,
            colorization_log,
            refinement_steps,
            sh_degree_n,
            n_channels,
        )
    else:
        raise ValueError(f"Unsupported color_opt '{color_opt}'.")

    save_colorization_outputs(
        scene_path,
        iteration,
        color_opt,
        lighting_index,
        SAVE_LOSS,
        loss_history,
        colorization_log,
        final_gaussians,
        output_path,
    )


def compute_image_metrics(
    image_chw: torch.Tensor, gt_chw: torch.Tensor, ssim_weight: float = 0.2
):
    """
    Compute standard image quality metrics for one rendered view.

    Both inputs are expected in CHW layout and normalized to [0, 1]. The
    returned dictionary contains the individual metrics as well as a combined
    reconstruction objective frequently used in Gaussian Splatting pipelines.
    """
    image = torch.clamp(image_chw, 0.0, 1.0)
    gt = torch.clamp(gt_chw, 0.0, 1.0)
    # Basic metrics
    l1 = l1_loss(image, gt).mean()
    l2 = F.mse_loss(image, gt).mean()
    ps = psnr(image, gt).mean()
    # SSIM expects NCHW
    ssim_val = msssim_ssim(
        image.unsqueeze(0), gt.unsqueeze(0), data_range=1.0, size_average=True
    )
    combined = l1 + (1.0 - ssim_val) * float(ssim_weight)
    return {
        "l1": l1,
        "l2": l2,
        "psnr": ps,
        "ssim": ssim_val,
        "loss_l1_ssim": combined,
    }


@torch.no_grad()
def compute_losses(
    gaussians, cam_list, pipeline_params, bg_color, ssim_weight: float = 0.2
):
    """Average image metrics over a list of evaluation cameras."""
    device = resolve_device()
    if len(cam_list) == 0:
        return {"l1": 0.0, "l2": 0.0, "psnr": 0.0, "ssim": 0.0, "loss_l1_ssim": 0.0}

    l1_sum = 0.0
    l2_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    loss_sum = 0.0
    for viewpoint in cam_list:
        rendered = render(viewpoint, gaussians, pipeline_params, bg_color)["render"]
        gt = viewpoint.original_image.to(device)
        m = compute_image_metrics(rendered, gt, ssim_weight=ssim_weight)
        l1_sum += float(m["l1"].item())
        l2_sum += float(m["l2"].item())
        psnr_sum += float(m["psnr"].item())
        ssim_sum += float(m["ssim"].item())
        loss_sum += float(m["loss_l1_ssim"].item())

    n = float(len(cam_list))
    return {
        "l1": l1_sum / n,
        "l2": l2_sum / n,
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
        "loss_l1_ssim": loss_sum / n,
    }


if __name__ == "__main__":
    parser = ArgumentParser("Assign color from gradients")
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to the model folder (output/...)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="Iteration of the saved model (e.g., 30000)",
    )
    parser.add_argument(
        "--color_opt",
        type=str,
        default="instant",
        choices=["instant", "adam"],
        help="Color optimization method",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        help="Optional: path to the source data (overrides automatic path)",
    )

    # additional parameters for the optimization process
    parser.add_argument(
        "--refinement_steps", type=int, default=50, help="Number of refinement steps"
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Spherical harmonics degree usually 3 and 0 for segmentation",
    )
    parser.add_argument(
        "--save_loss",
        type=bool,
        default=False,
        help="Save and print train/test metrics",
    )
    parser.add_argument(
        "--plot_images", type=bool, default=False, help="Plot rendered vs GT images"
    )
    parser.add_argument(
        "--lighting_index", type=int, default=0, help="Lighting condition index"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: path and name of the output file",
    )


    args = parser.parse_args()
    colorize(
        args.scene_path,
        args.iteration,
        args.color_opt,
        args.source_path,
        args.refinement_steps,
        args.sh_degree,
        args.save_loss,
        args.plot_images,
        args.lighting_index,
        args.output_path,
    )