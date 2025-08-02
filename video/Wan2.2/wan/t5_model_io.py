def sanitize(weights):
    """Following the pattern used in MLX Stable Diffusion."""
    new_weights = {}

    for key, weight in weights.items():
        # Handle the main structural difference: FFN gate layer
        if ".ffn.gate.0.weight" in key:
            # PyTorch has Sequential(Linear, GELU) but MLX has separate gate_proj + gate_act
            key = key.replace(".ffn.gate.0.weight", ".ffn.gate_proj.weight")

        elif ".ffn.gate.0.bias" in key:
            # Handle bias if it exists
            key = key.replace(".ffn.gate.0.bias", ".ffn.gate_proj.bias")

        elif ".ffn.gate.1" in key:
            # Skip GELU activation parameters - MLX handles this separately
            print(f"Skipping GELU parameter: {key}")
            continue

        # Handle any other potential FFN mappings
        elif ".ffn.fc1.weight" in key:
            pass
        elif ".ffn.fc2.weight" in key:
            pass

        # Handle attention layers (should be direct mapping)
        elif ".attn.q.weight" in key:
            pass
        elif ".attn.k.weight" in key:
            pass
        elif ".attn.v.weight" in key:
            pass
        elif ".attn.o.weight" in key:
            pass

        # Handle embeddings and norms (direct mapping)
        elif "token_embedding.weight" in key:
            pass
        elif "pos_embedding.embedding.weight" in key:
            pass
        elif "norm1.weight" in key or "norm2.weight" in key or "norm.weight" in key:
            pass

        # Default: direct mapping for any other parameters
        else:
            pass

        new_weights[key] = weight

    return new_weights
