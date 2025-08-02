import os

import torch
from safetensors.torch import save_file


def convert_pickle_to_safetensors(
        pickle_path: str,
        safetensors_path: str,
        load_method: str = "weights_only"  # Changed default to weights_only
):
    """Convert PyTorch pickle file to safetensors format."""

    print(f"Loading PyTorch weights from: {pickle_path}")

    # Try multiple loading methods in order of preference
    methods_to_try = ["weights_only", "state_dict", "full_model"]
    methods_to_try.remove(load_method)
    methods_to_try.insert(0, load_method)

    for method in methods_to_try:
        try:
            if method == "weights_only":
                state_dict = torch.load(pickle_path, map_location='cpu', weights_only=True)

            elif method == "state_dict":
                checkpoint = torch.load(pickle_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

            elif method == "full_model":
                model = torch.load(pickle_path, map_location='cpu')
                if hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
                else:
                    state_dict = model

            print(f"✅ Successfully loaded with method: {method}")
            break

        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    else:
        raise RuntimeError(f"All loading methods failed for {pickle_path}")

    # Clean up the state dict
    state_dict = clean_state_dict(state_dict)

    print(f"Found {len(state_dict)} parameters")

    # Save as safetensors
    print(f"Saving to safetensors: {safetensors_path}")
    os.makedirs(os.path.dirname(safetensors_path), exist_ok=True)
    save_file(state_dict, safetensors_path)

    print("✅ Conversion complete!")
    return state_dict


def clean_state_dict(state_dict):
    """
    Clean up state dict by removing unwanted prefixes or keys.
    """
    cleaned = {}

    for key, value in state_dict.items():
        # Remove common prefixes that might interfere
        clean_key = key

        if clean_key.startswith('module.'):
            clean_key = clean_key[7:]

        if clean_key != key:
            print(f"Cleaned key: {key} -> {clean_key}")

        cleaned[clean_key] = value

    return cleaned
