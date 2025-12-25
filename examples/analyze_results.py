"""
Spectra Vis - Analyze Results Example

This example demonstrates how to load and analyze captured
tensor data without running the API server.

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra_vis.model_profiles import (
    get_model_profile,
    get_layer_count,
    map_observation_to_layer,
    TENSOR_COLORS,
)


def load_capture_file(filepath: str) -> dict:
    """Load a capture JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_capture(data: dict, model_family: str, model_size: str = "7b"):
    """Analyze captured tensor data."""

    print("=" * 60)
    print("Spectra Vis - Results Analysis")
    print("=" * 60)
    print()

    # Basic info
    print("Capture Info:")
    print(f"  Tool: {data.get('tool', 'Unknown')}")
    print(f"  Created: {data.get('created', 'Unknown')}")
    print(f"  Model Family: {data.get('model_family', 'Unknown')}")
    print()

    # Get model profile
    profile = get_model_profile(model_family)
    if not profile:
        print(f"Warning: Unknown model family '{model_family}'")
        return

    total_layers = get_layer_count(model_family, model_size)
    print(f"Model Profile: {profile.get('description', model_family)}")
    print(f"Layer Count: {total_layers}")
    print()

    # Statistics
    stats = data.get("statistics", {})
    print("Statistics:")
    print(f"  Total trials: {stats.get('total_trials', len(data.get('trials', [])))}")
    print(f"  Unique tensors: {stats.get('unique_tensors', 0)}")
    print(f"  Total callbacks: {stats.get('total_callbacks', 0)}")
    print()

    # Trial summaries
    trials = data.get("trials", [])
    if trials:
        print("Trial Summary:")
        print("-" * 60)
        for trial in trials:
            print(f"  Trial {trial.get('trial_id', '?')}:")
            print(f"    Prompt: {trial.get('prompt', '')[:50]}...")
            print(f"    Response: {trial.get('response', '')[:50]}...")
            print(f"    Inference: {trial.get('inference_time_ms', 0):.1f}ms")
            print(f"    Tensors: {len(trial.get('tensor_counts', {}))}")
        print()

    # Map observations to layers
    tensor_order = data.get("tensor_order", [])
    if not tensor_order and trials:
        # Use first trial's order
        tensor_order = trials[0].get("tensor_order", [])

    # Aggregate counts across all trials
    total_counts = Counter()
    for trial in trials:
        for addr, count in trial.get("tensor_counts", {}).items():
            total_counts[addr] += count

    print("Layer Analysis:")
    print("-" * 60)

    # Group by layer
    layer_stats = {}
    for i, addr in enumerate(tensor_order):
        mapping = map_observation_to_layer(i, model_family, total_layers)
        layer = mapping["layer"]
        tensor_type = mapping["type"]
        count = total_counts.get(addr, 0)

        if layer not in layer_stats:
            layer_stats[layer] = {
                "tensors": [],
                "total_count": 0,
            }

        layer_stats[layer]["tensors"].append({
            "type": tensor_type,
            "count": count,
        })
        layer_stats[layer]["total_count"] += count

    # Print layer stats (first few and last few)
    sorted_layers = sorted(layer_stats.keys())

    # Special layers
    special = [l for l in sorted_layers if l < 0 or l >= 900]
    regular = [l for l in sorted_layers if 0 <= l < 900]

    # Print embedding
    if -1 in layer_stats:
        ls = layer_stats[-1]
        print(f"  Embedding (Layer -1):")
        print(f"    Tensors: {len(ls['tensors'])}")
        print(f"    Total observations: {ls['total_count']}")

    # Print first few regular layers
    for layer in regular[:3]:
        ls = layer_stats[layer]
        print(f"  Layer {layer}:")
        print(f"    Tensors: {len(ls['tensors'])}")
        print(f"    Total observations: {ls['total_count']}")
        types = [t["type"] for t in ls["tensors"]]
        print(f"    Types: {', '.join(types)}")

    if len(regular) > 6:
        print(f"  ... ({len(regular) - 6} more layers) ...")

    # Print last few regular layers
    for layer in regular[-3:]:
        if layer not in regular[:3]:  # Don't repeat
            ls = layer_stats[layer]
            print(f"  Layer {layer}:")
            print(f"    Tensors: {len(ls['tensors'])}")
            print(f"    Total observations: {ls['total_count']}")

    # Print output layers
    if 998 in layer_stats:
        ls = layer_stats[998]
        print(f"  Output Norm (Layer 998):")
        print(f"    Total observations: {ls['total_count']}")

    if 999 in layer_stats:
        ls = layer_stats[999]
        print(f"  Output (Layer 999):")
        print(f"    Total observations: {ls['total_count']}")

    print()

    # Category breakdown
    categories = profile.get("tensor_categories", {})
    print("Category Breakdown:")
    print("-" * 60)

    category_counts = {cat: 0 for cat in categories}
    category_counts["unknown"] = 0

    for i, addr in enumerate(tensor_order):
        mapping = map_observation_to_layer(i, model_family, total_layers)
        tensor_type = mapping["type"]
        count = total_counts.get(addr, 0)

        found = False
        for cat, types in categories.items():
            if tensor_type in types:
                category_counts[cat] += count
                found = True
                break
        if not found:
            category_counts["unknown"] += count

    total = sum(category_counts.values())
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / total * 100) if total > 0 else 0
            color = TENSOR_COLORS.get(cat, "#808080")
            print(f"  {cat:15} {count:8} observations ({pct:5.1f}%) {color}")

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main():
    """Main entry point."""

    # Default file path (update this or pass as argument)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "capture_data.json"

    # Model settings (update as needed)
    MODEL_FAMILY = "llama"
    MODEL_SIZE = "7b"

    try:
        data = load_capture_file(filepath)
        analyze_capture(data, MODEL_FAMILY, MODEL_SIZE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("Usage: python analyze_results.py [capture_file.json]")
        print()
        print("If no file is specified, looks for 'capture_data.json'")
        sys.exit(1)


if __name__ == "__main__":
    main()
