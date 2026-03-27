import argparse
import torch
from scene.gaussian_model import GaussianModel


def combine_ply(ply_paths, weights, output_path):
    # Load gaussians
    gaussians = []
    for ply_path in ply_paths:
        gm = GaussianModel(sh_degree=3)
        gm.load_ply(ply_path)
        gaussians.append(gm)

    # Combine gaussians
    combined_gaussians = gaussians[-1].clone()
    combined_gaussians._features_dc.data[:] = 0
    combined_gaussians._features_rest.data[:] = 0

    for gaussian, weight in zip(gaussians, weights):
        if isinstance(weight, float):
            weight = [weight, weight, weight]
        if isinstance(weight, list):
            weight = (
                torch.tensor(
                    weight,
                    dtype=gaussian._features_dc.dtype,
                    device=gaussian._features_dc.device,
                )
                .unsqueeze(0)
                .unsqueeze(1)
            )

        combined_gaussians._features_dc.data += gaussian._features_dc * weight
        combined_gaussians._features_rest.data += gaussian._features_rest * weight

    # Save combined gaussians
    combined_gaussians.save_ply(output_path)


def mask_gaussians(gaussians, mask):
    """
    Filters a GaussianModel and returns a new model
    containing only the Gaussians selected by mask.
    """
    clamped = gaussians.clone()

    clamped._xyz = gaussians._xyz[mask].clone()
    clamped._features_dc = gaussians._features_dc[mask].clone()
    clamped._features_rest = gaussians._features_rest[mask].clone()
    clamped._scaling = gaussians._scaling[mask].clone()
    clamped._rotation = gaussians._rotation[mask].clone()
    clamped._opacity = gaussians._opacity[mask].clone()

    return clamped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask Gaussians from a source PLY using a mask PLY."
    )

    parser.add_argument(
        "--source_file",
        type=str,
        required=True,
        help="Path to the source point cloud PLY file.",
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        required=True,
        help="Path to the mask point cloud PLY file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path where the masked output PLY should be saved.",
    )
    parser.add_argument(
        "--source_sh_degree",
        type=int,
        default=3,
        help="SH degree for the source GaussianModel (default: 3).",
    )
    parser.add_argument(
        "--mask_sh_degree",
        type=int,
        default=0,
        help="SH degree for the mask GaussianModel (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold for mask selection on features_dc (default: 0.6).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    source_gaussians = GaussianModel(sh_degree=args.source_sh_degree)
    source_gaussians.load_ply(args.source_file)

    masked_gaussians = GaussianModel(sh_degree=args.mask_sh_degree)
    masked_gaussians.load_ply(args.mask_file)

    mask = torch.any(masked_gaussians._features_dc[:, 0] > args.threshold, dim=1)

    output_gaussians = mask_gaussians(source_gaussians, mask)
    output_gaussians.save_ply(args.output_file)
