"""
Spectra Vis - Backend Server

FastAPI server for serving tensor visualization data.
Provides REST API endpoints for loading captured data and
serving model profiles to the frontend visualization.

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra_vis.model_profiles import (
    get_model_profile,
    get_supported_models,
    get_layer_count,
    map_observation_to_layer,
    TENSOR_COLORS,
    FLOW_COLORS,
    TENSOR_FLOWS,
)

# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Spectra Vis API",
    description="Tensor activation visualization for GGUF models",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Data Storage
# -----------------------------------------------------------------------------

# In-memory storage for loaded data
_loaded_data: Optional[Dict[str, Any]] = None
_data_path: Optional[str] = None

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------

class LoadRequest(BaseModel):
    """Request to load a capture file."""
    filepath: str


class AnalyzeRequest(BaseModel):
    """Request to analyze loaded data."""
    model_family: str
    model_size: str = "7b"
    trial_id: Optional[int] = None


class TensorGridRequest(BaseModel):
    """Request for tensor grid data."""
    model_family: str
    model_size: str = "7b"
    trial_id: Optional[int] = None


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Spectra Vis API",
        "version": "1.0.0",
        "author": "3rdEyeVisuals",
        "endpoints": [
            "/api/models",
            "/api/files",
            "/api/load",
            "/api/status",
            "/api/analyze",
            "/api/tensor-grid",
            "/api/colors",
            "/api/flows",
        ]
    }


@app.get("/api/models")
async def get_models():
    """Get list of supported model families."""
    models = get_supported_models()
    profiles = {}
    for model in models:
        profile = get_model_profile(model)
        if profile:
            profiles[model] = {
                "description": profile.get("description", ""),
                "variants": profile.get("variants", []),
                "layer_counts": profile.get("layer_counts", {}),
            }
    return {
        "supported_models": models,
        "profiles": profiles,
    }


@app.get("/api/files")
async def list_files():
    """List available capture files in the data directory."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return {"files": [], "directory": str(DATA_DIR)}

    files = []
    for f in DATA_DIR.glob("*.json"):
        try:
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
        except Exception:
            continue

    # Sort by modification time, newest first
    files.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "files": files,
        "directory": str(DATA_DIR),
    }


@app.post("/api/load")
async def load_file(request: LoadRequest):
    """Load a capture file for analysis."""
    global _loaded_data, _data_path

    filepath = Path(request.filepath)

    # Security: only allow loading from data directory or absolute paths
    if not filepath.is_absolute():
        filepath = DATA_DIR / filepath

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    if not filepath.suffix == ".json":
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Validate structure
    if "tensor_order" not in data and "trials" not in data:
        raise HTTPException(
            status_code=400,
            detail="Invalid capture file: missing tensor_order or trials"
        )

    _loaded_data = data
    _data_path = str(filepath)

    return {
        "status": "loaded",
        "filepath": str(filepath),
        "model_family": data.get("model_family", "unknown"),
        "trials": len(data.get("trials", [])),
        "unique_tensors": len(data.get("tensor_order", [])),
        "statistics": data.get("statistics", {}),
    }


@app.get("/api/status")
async def get_status():
    """Get current server status and loaded data info."""
    if _loaded_data is None:
        return {
            "status": "ready",
            "loaded": False,
            "message": "No data loaded. Use /api/load to load a capture file.",
        }

    return {
        "status": "ready",
        "loaded": True,
        "filepath": _data_path,
        "model_family": _loaded_data.get("model_family", "unknown"),
        "trials": len(_loaded_data.get("trials", [])),
        "unique_tensors": len(_loaded_data.get("tensor_order", [])),
    }


@app.post("/api/analyze")
async def analyze_data(request: AnalyzeRequest):
    """Analyze loaded data and map observations to layers."""
    if _loaded_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    model_family = request.model_family
    model_size = request.model_size
    total_layers = get_layer_count(model_family, model_size)

    profile = get_model_profile(model_family)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model family: {model_family}"
        )

    # Get tensor order from the specified trial or global
    if request.trial_id is not None:
        trials = _loaded_data.get("trials", [])
        if request.trial_id >= len(trials):
            raise HTTPException(
                status_code=400,
                detail=f"Trial {request.trial_id} not found"
            )
        tensor_order = trials[request.trial_id].get("tensor_order", [])
        tensor_counts = trials[request.trial_id].get("tensor_counts", {})
    else:
        tensor_order = _loaded_data.get("tensor_order", [])
        tensor_counts = {}
        for trial in _loaded_data.get("trials", []):
            for addr, count in trial.get("tensor_counts", {}).items():
                tensor_counts[addr] = tensor_counts.get(addr, 0) + count

    # Map each observation to layer and type
    mapped_tensors = []
    for i, addr in enumerate(tensor_order):
        mapping = map_observation_to_layer(i, model_family, total_layers)
        mapped_tensors.append({
            "address": addr,
            "observation_index": i,
            "layer": mapping["layer"],
            "type": mapping["type"],
            "count": tensor_counts.get(addr, 0),
        })

    # Group by layer
    layers = {}
    for t in mapped_tensors:
        layer = t["layer"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(t)

    return {
        "model_family": model_family,
        "model_size": model_size,
        "total_layers": total_layers,
        "total_tensors": len(mapped_tensors),
        "tensors": mapped_tensors,
        "layers": layers,
        "categories": profile.get("tensor_categories", {}),
    }


@app.post("/api/tensor-grid")
async def get_tensor_grid(request: TensorGridRequest):
    """Get tensor data formatted for 3D grid visualization."""
    if _loaded_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    model_family = request.model_family
    model_size = request.model_size
    total_layers = get_layer_count(model_family, model_size)

    profile = get_model_profile(model_family)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model family: {model_family}"
        )

    tensor_types = profile.get("tensor_types", [])
    categories = profile.get("tensor_categories", {})

    # Get tensor data
    if request.trial_id is not None:
        trials = _loaded_data.get("trials", [])
        if request.trial_id >= len(trials):
            raise HTTPException(
                status_code=400,
                detail=f"Trial {request.trial_id} not found"
            )
        tensor_order = trials[request.trial_id].get("tensor_order", [])
        tensor_counts = trials[request.trial_id].get("tensor_counts", {})
    else:
        tensor_order = _loaded_data.get("tensor_order", [])
        tensor_counts = {}
        for trial in _loaded_data.get("trials", []):
            for addr, count in trial.get("tensor_counts", {}).items():
                tensor_counts[addr] = tensor_counts.get(addr, 0) + count

    # Build grid data structure
    grid = []

    # Find max count for normalization
    max_count = max(tensor_counts.values()) if tensor_counts else 1

    # Calculate actual tensors per layer based on data
    num_tensors = len(tensor_order)
    # Use actual layer count from model, distribute tensors evenly
    tensors_per_layer = max(1, num_tensors // total_layers)

    # Color cycle for visual variety within categories
    category_colors = {
        "embedding": "#4a90d9",     # Blue
        "attention": "#50c878",     # Green
        "feedforward": "#ff7f50",   # Coral/Orange
        "output": "#da70d6",        # Purple
        "unknown": "#808080",       # Gray
    }

    for i, addr in enumerate(tensor_order):
        count = tensor_counts.get(addr, 0)

        # Calculate layer based on even distribution
        if i == 0:
            layer = -1  # First tensor is embedding
            tensor_type = "token_embd"
            category = "embedding"
        elif i >= num_tensors - 2:
            # Last two tensors are output
            if i == num_tensors - 2:
                layer = 998
                tensor_type = "output_norm"
                category = "output"
            else:
                layer = 999
                tensor_type = "output"
                category = "output"
        else:
            # Distribute remaining tensors across layers
            adjusted_idx = i - 1
            layer = adjusted_idx // tensors_per_layer
            type_idx = adjusted_idx % tensors_per_layer

            # Cap layer at total_layers - 1
            if layer >= total_layers:
                layer = total_layers - 1

            # Assign type based on position within layer
            if tensors_per_layer > 0 and len(tensor_types) > 0:
                # Map type_idx to tensor_types cyclically
                type_idx_mapped = type_idx % len(tensor_types)
                tensor_type = tensor_types[type_idx_mapped]
            else:
                tensor_type = f"tensor_{type_idx}"

            # Determine category from tensor type
            category = "unknown"
            for cat, types in categories.items():
                if tensor_type in types:
                    category = cat
                    break

        grid.append({
            "address": addr,
            "layer": layer,
            "type": tensor_type,
            "type_index": i % tensors_per_layer if i > 0 and i < num_tensors - 2 else 0,
            "category": category,
            "count": count,
            "intensity": count / max_count if max_count > 0 else 0,
            "color": category_colors.get(category, category_colors["unknown"]),
        })

    return {
        "model_family": model_family,
        "model_size": model_size,
        "total_layers": total_layers,
        "tensors_per_layer": tensors_per_layer,
        "tensor_types": tensor_types,
        "grid": grid,
        "max_count": max_count,
        "colors": category_colors,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a capture file to the data directory."""
    # Validate file type
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '-', '.'))
    if not safe_filename:
        safe_filename = "upload.json"

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Read and validate JSON content
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Validate structure
    if "tensor_order" not in data and "trials" not in data:
        raise HTTPException(
            status_code=400,
            detail="Invalid capture file: missing tensor_order or trials"
        )

    # Save to data directory
    filepath = DATA_DIR / safe_filename

    # Avoid overwriting - add number suffix if exists
    counter = 1
    original_stem = filepath.stem
    while filepath.exists():
        filepath = DATA_DIR / f"{original_stem}_{counter}.json"
        counter += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return {
        "status": "uploaded",
        "filename": filepath.name,
        "filepath": str(filepath),
        "size_bytes": len(content),
    }


@app.get("/api/colors")
async def get_colors():
    """Get color scheme definitions."""
    return {
        "tensor_colors": TENSOR_COLORS,
        "flow_colors": FLOW_COLORS,
    }


@app.get("/api/flows")
async def get_flows():
    """Get tensor flow definitions for arrow rendering."""
    return {
        "flows": TENSOR_FLOWS,
        "colors": FLOW_COLORS,
    }


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    """Run the server."""
    import uvicorn
    print("=" * 60)
    print("Spectra Vis API Server")
    print("Copyright (c) 2025 3rdEyeVisuals")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
