"""
Spectra Vis - Model Profiles

Pre-configured tensor layout profiles for supported model architectures.
Users select their model family, and this module provides the expected
tensor structure for visualization mapping.

Tensor information is based on publicly documented transformer architectures
and the GGUF format specification.

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

from typing import Dict, List, Any, Optional


# -----------------------------------------------------------------------------
# Supported Model Families
# -----------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "llama",
    "granite",
    "qwen",
    "phi",
    "mistral",
]


# -----------------------------------------------------------------------------
# Model Profile Definitions
# -----------------------------------------------------------------------------
# Each profile defines:
# - description: Human-readable description
# - layer_count: Typical number of layers (varies by model size)
# - tensor_types: List of tensor types in execution order within each layer
# - tensor_categories: Grouping for visualization (attention, ffn, norm, etc.)

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {

    "llama": {
        "description": "Llama / Llama 2 / Llama 3 (Meta)",
        "variants": ["llama-7b", "llama-13b", "llama-2-7b", "llama-2-13b",
                     "llama-3-8b", "llama-3.1-8b", "llama-3.2-1b", "llama-3.2-3b"],
        "layer_counts": {
            "1b": 16,
            "3b": 28,
            "7b": 32,
            "8b": 32,
            "13b": 40,
            "70b": 80,
        },
        "tensor_types": [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        "special_tensors": {
            "token_embd": -1,       # Embedding layer (before layer 0)
            "output_norm": 998,     # Final normalization
            "output": 999,          # Output projection / logits
        },
        "tensor_categories": {
            "embedding": ["token_embd"],
            "attention": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output"],
            "feedforward": ["ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
            "output": ["output_norm", "output"],
        },
    },

    "granite": {
        "description": "Granite (IBM)",
        "variants": ["granite-3b", "granite-8b", "granite-20b",
                     "granite-3-2b", "granite-3-8b"],
        "layer_counts": {
            "2b": 24,
            "3b": 32,
            "8b": 32,
            "20b": 52,
        },
        "tensor_types": [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        "special_tensors": {
            "token_embd": -1,       # Embedding layer (before layer 0)
            "output_norm": 998,     # Final normalization
            "output": 999,          # Output projection / logits
        },
        "tensor_categories": {
            "embedding": ["token_embd"],
            "attention": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output"],
            "feedforward": ["ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
            "output": ["output_norm", "output"],
        },
    },

    "qwen": {
        "description": "Qwen / Qwen 2 (Alibaba)",
        "variants": ["qwen-7b", "qwen-14b", "qwen2-0.5b", "qwen2-1.5b",
                     "qwen2-7b", "qwen2.5-7b", "qwen2.5-14b"],
        "layer_counts": {
            "0.5b": 24,
            "1.5b": 28,
            "7b": 32,
            "14b": 40,
            "72b": 80,
        },
        "tensor_types": [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        "special_tensors": {
            "token_embd": -1,       # Embedding layer (before layer 0)
            "output_norm": 998,     # Final normalization
            "output": 999,          # Output projection / logits
        },
        "tensor_categories": {
            "embedding": ["token_embd"],
            "attention": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output"],
            "feedforward": ["ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
            "output": ["output_norm", "output"],
        },
    },

    "phi": {
        "description": "Phi / Phi-2 / Phi-3 (Microsoft)",
        "variants": ["phi-2", "phi-3-mini", "phi-3-small", "phi-3-medium"],
        "layer_counts": {
            "phi-2": 32,
            "mini": 32,
            "small": 32,
            "medium": 40,
        },
        "tensor_types": [
            "attn_norm",
            "attn_qkv",
            "attn_output",
            "ffn_norm",
            "ffn_up",
            "ffn_down",
        ],
        "special_tensors": {
            "token_embd": -1,       # Embedding layer (before layer 0)
            "output_norm": 998,     # Final normalization
            "output": 999,          # Output projection / logits
        },
        "tensor_categories": {
            "embedding": ["token_embd"],
            "attention": ["attn_norm", "attn_qkv", "attn_output"],
            "feedforward": ["ffn_norm", "ffn_up", "ffn_down"],
            "output": ["output_norm", "output"],
        },
        "notes": "Phi uses fused QKV projection (attn_qkv) instead of separate Q, K, V",
    },

    "mistral": {
        "description": "Mistral / Mixtral (Mistral AI)",
        "variants": ["mistral-7b", "mixtral-8x7b", "mixtral-8x22b"],
        "layer_counts": {
            "7b": 32,
            "8x7b": 32,
            "8x22b": 56,
        },
        "tensor_types": [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        "special_tensors": {
            "token_embd": -1,       # Embedding layer (before layer 0)
            "output_norm": 998,     # Final normalization
            "output": 999,          # Output projection / logits
        },
        "tensor_categories": {
            "embedding": ["token_embd"],
            "attention": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output"],
            "feedforward": ["ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
            "output": ["output_norm", "output"],
        },
        "notes": "Mixtral variants use Mixture of Experts (MoE) in FFN layers",
    },
}


# -----------------------------------------------------------------------------
# Tensor Flow Definitions
# -----------------------------------------------------------------------------
# Defines how tensors connect to each other within a layer.
# Used for rendering flow arrows in the visualization.

TENSOR_FLOWS = {
    # Attention block flow
    "attention_flow": [
        ("token_embd", "attn_norm"),
        ("attn_norm", "attn_q"),
        ("attn_norm", "attn_k"),
        ("attn_norm", "attn_v"),
        ("attn_q", "attn_output"),
        ("attn_k", "attn_output"),
        ("attn_v", "attn_output"),
    ],

    # FFN block flow
    "ffn_flow": [
        ("attn_output", "ffn_norm"),
        ("ffn_norm", "ffn_gate"),
        ("ffn_norm", "ffn_up"),
        ("ffn_gate", "ffn_down"),
        ("ffn_up", "ffn_down"),
    ],

    # Cross-layer flow (output of one layer to input of next)
    "layer_flow": [
        ("ffn_down", "attn_norm"),  # Residual connection to next layer
    ],

    # Output flow
    "output_flow": [
        ("ffn_down", "output_norm"),
        ("output_norm", "output"),
    ],
}


# -----------------------------------------------------------------------------
# Visualization Colors
# -----------------------------------------------------------------------------
# Color scheme for tensor categories in the visualization.

TENSOR_COLORS = {
    "embedding": "#4a90d9",     # Blue
    "attention": "#50c878",     # Green
    "feedforward": "#ff7f50",   # Coral/Orange
    "output": "#da70d6",        # Orchid/Purple
    "norm": "#ffd700",          # Gold
    "unknown": "#808080",       # Gray
}

FLOW_COLORS = {
    "attention_flow": "#50c878",
    "ffn_flow": "#ff7f50",
    "layer_flow": "#ffffff",
    "output_flow": "#da70d6",
}


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------

def get_model_profile(model_family: str) -> Optional[Dict[str, Any]]:
    """
    Get the tensor profile for a model family.

    Args:
        model_family: Model family name (e.g., "llama", "granite")

    Returns:
        Profile dict with tensor structure info, or None if not found
    """
    family = model_family.lower().strip()
    return MODEL_PROFILES.get(family)


def get_supported_models() -> List[str]:
    """
    Get list of supported model families.

    Returns:
        List of model family names
    """
    return SUPPORTED_MODELS.copy()


def get_tensor_categories(model_family: str) -> Dict[str, List[str]]:
    """
    Get tensor categories for a model family.

    Args:
        model_family: Model family name

    Returns:
        Dict mapping category name to list of tensor types
    """
    profile = get_model_profile(model_family)
    if profile:
        return profile.get("tensor_categories", {})
    return {}


def get_layer_count(model_family: str, model_size: str) -> int:
    """
    Get the layer count for a specific model variant.

    Args:
        model_family: Model family name (e.g., "llama")
        model_size: Model size identifier (e.g., "7b", "8b")

    Returns:
        Number of layers, or 32 as default
    """
    profile = get_model_profile(model_family)
    if profile:
        layer_counts = profile.get("layer_counts", {})
        size = model_size.lower().strip()
        if size in layer_counts:
            return layer_counts[size]
    return 32  # Default


def estimate_tensors_per_layer(model_family: str) -> int:
    """
    Estimate how many tensors are evaluated per layer.

    Useful for mapping observation order to layer index.

    Args:
        model_family: Model family name

    Returns:
        Estimated tensor count per layer
    """
    profile = get_model_profile(model_family)
    if profile:
        return len(profile.get("tensor_types", []))
    return 9  # Default (typical transformer layer)


def map_observation_to_layer(
    observation_index: int,
    model_family: str,
    total_layers: int
) -> Dict[str, Any]:
    """
    Map a tensor observation index to its likely layer and type.

    Based on the observation order during inference, estimate which
    layer and tensor type this observation corresponds to.

    Layer numbering convention:
    - Layer -1: Embedding (token_embd)
    - Layers 0 to (total_layers - 1): Transformer blocks
    - Layer 998: Final normalization (output_norm)
    - Layer 999: Output projection (output/logits)

    Args:
        observation_index: Index in the observation order (0-based)
        model_family: Model family name
        total_layers: Total number of layers in the model

    Returns:
        Dict with estimated layer index and tensor type
    """
    profile = get_model_profile(model_family)
    if not profile:
        return {"layer": -1, "type": "unknown"}

    tensor_types = profile.get("tensor_types", [])
    special_tensors = profile.get("special_tensors", {})
    tensors_per_layer = len(tensor_types)

    # First observation is typically the embedding
    if observation_index == 0:
        return {"layer": -1, "type": "token_embd"}

    # Account for embedding tensor at start
    adjusted_index = observation_index - 1

    # Calculate which layer and tensor type within that layer
    layer = adjusted_index // tensors_per_layer
    type_index = adjusted_index % tensors_per_layer

    # Check if we're past the main transformer layers
    if layer >= total_layers:
        # Output tensors
        extra_index = adjusted_index - (total_layers * tensors_per_layer)
        if extra_index == 0:
            return {"layer": 998, "type": "output_norm"}
        else:
            return {"layer": 999, "type": "output"}

    # Return the layer and tensor type
    if type_index < len(tensor_types):
        return {"layer": layer, "type": tensor_types[type_index]}

    return {"layer": layer, "type": "unknown"}
