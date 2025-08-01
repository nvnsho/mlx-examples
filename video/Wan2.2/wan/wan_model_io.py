# wan_model_io.py

import glob
import os
from typing import Tuple, Set, Dict

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten


def sanitize(weights):
    """Sanitize PyTorch WAN 2.2 weights to adapt MLX format."""

    # Only add .layers to Sequential WITHIN components, not to blocks themselves
    # blocks.N stays as blocks.N (not blocks.layers.N)

    # Handle Sequential layers - PyTorch uses .0, .1, .2, MLX uses .layers.0, .layers.1, .layers.2
    # Only for components INSIDE blocks and top-level modules

    new_weights = {}

    for key, weight in weights.items():
        if ".ffn." in key and not ".layers." in key:
            # Replace .ffn.0 with .ffn.layers.0, etc.
            key = key.replace(".ffn.0.", ".ffn.layers.0.")
            key = key.replace(".ffn.1.", ".ffn.layers.1.")
            key = key.replace(".ffn.2.", ".ffn.layers.2.")

        elif "text_embedding." in key and not ".layers." in key:
            for i in range(10):
                key = key.replace(f"text_embedding.{i}.", f"text_embedding.layers.{i}.")

        elif "time_embedding." in key and not ".layers." in key:
            for i in range(10):
                key = key.replace(f"time_embedding.{i}.", f"time_embedding.layers.{i}.")

        elif "time_projection." in key and not ".layers." in key:
            for i in range(10):
                key = key.replace(f"time_projection.{i}.", f"time_projection.layers.{i}.")

        # Handle conv transpose for patch_embedding
        elif "patch_embedding.weight" in key:
            # PyTorch Conv3d: (out_channels, in_channels, D, H, W)
            # MLX Conv3d: (out_channels, D, H, W, in_channels)
            weight = mx.transpose(weight, (0, 2, 3, 4, 1))

        new_weights[key] = weight

    return new_weights


def check_parameter_mismatch(model, weights: Dict[str, mx.array]) -> Tuple[Set[str], Set[str]]:
    """
    Check for parameter mismatches between model and weights.

    Returns:
        (model_only, weights_only): Sets of parameter names that exist only in model or weights
    """
    # Get all parameter names from model
    model_params = dict(tree_flatten(model.parameters()))
    model_keys = set(model_params.keys())

    # Remove computed buffers that aren't loaded from weights
    computed_buffers = {'freqs'}  # Add any other computed buffers here
    model_keys = model_keys - computed_buffers

    # Get all parameter names from weights
    weight_keys = set(weights.keys())

    # Find differences
    model_only = model_keys - weight_keys
    weights_only = weight_keys - model_keys

    return model_only, weights_only


def load_wan_2_2_from_safetensors(
    safetensors_path: str,
    model,
    dtype=mx.float32,
    check_mismatch: bool = True
):
    """
    Load WAN 2.2 Model weights from safetensors file(s) into MLX model.

    Args:
        safetensors_path: Path to safetensors file or directory
        model: MLX model instance
        float16: Whether to use float16 precision
        check_mismatch: Whether to check for parameter mismatches
    """
    if os.path.isdir(safetensors_path):
        # Multiple files (14B model) - only diffusion_pytorch_model files
        pattern = os.path.join(safetensors_path, "diffusion_pytorch_model*.safetensors")
        safetensor_files = sorted(glob.glob(pattern))
        print(f"Found {len(safetensor_files)} diffusion_pytorch_model safetensors files")

        # Load all files and merge weights
        all_weights = {}
        for file_path in safetensor_files:
            print(f"Loading: {file_path}")
            weights = mx.load(file_path)
            all_weights.update(weights)

        for key, weight in all_weights.items():
            if weight.dtype != dtype and mx.issubdtype(weight.dtype, mx.floating):
                all_weights[key] = weight.astype(dtype)

        all_weights = sanitize(all_weights)

        if check_mismatch:
            model_only, weights_only = check_parameter_mismatch(model, all_weights)

            if model_only:
                print(f"\n⚠️  WARNING: {len(model_only)} parameters in model but NOT in weights:")
                for param in sorted(model_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(model_only) > 10:
                    print(f"  ... and {len(model_only) - 10} more")

            if weights_only:
                print(f"\n⚠️  WARNING: {len(weights_only)} parameters in weights but NOT in model:")
                for param in sorted(weights_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(weights_only) > 10:
                    print(f"  ... and {len(weights_only) - 10} more")

            if not model_only and not weights_only:
                print("\n✅ Perfect match: All parameters align between model and weights!")

        model.update(tree_unflatten(list(all_weights.items())))
    else:
        # Single file
        print(f"Loading single file: {safetensors_path}")
        weights = mx.load(safetensors_path)

        for key, weight in weights.items():
            if weight.dtype != dtype and mx.issubdtype(weight.dtype, mx.floating):
                weights[key] = weight.astype(dtype)

        weights = sanitize(weights)

        if check_mismatch:
            model_only, weights_only = check_parameter_mismatch(model, weights)

            if model_only:
                print(f"\n⚠️  WARNING: {len(model_only)} parameters in model but NOT in weights:")
                for param in sorted(model_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(model_only) > 10:
                    print(f"  ... and {len(model_only) - 10} more")

            if weights_only:
                print(f"\n⚠️  WARNING: {len(weights_only)} parameters in weights but NOT in model:")
                for param in sorted(weights_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(weights_only) > 10:
                    print(f"  ... and {len(weights_only) - 10} more")

            if not model_only and not weights_only:
                print("\n✅ Perfect match: All parameters align between model and weights!")

        model.update(tree_unflatten(list(weights.items())))

    print("\nWAN 2.2 Model weights loaded successfully!")
    return model
