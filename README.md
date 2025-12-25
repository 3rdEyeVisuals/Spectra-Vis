# Spectra Vis (Tensor Edition)

**Neural Network Tensor Activation Visualization for GGUF Models**

By [3rdEyeVisuals](https://github.com/3rdEyeVisuals)

---

## Overview

Spectra Vis is an open-source tool for observing and visualizing tensor activations during Large Language Model (LLM) inference. It captures real-time tensor evaluation data from GGUF models running via `llama-cpp-python` and maps the observations to a visual representation of the model's architecture.

### Key Features

- **Tensor Observation**: Capture tensor activation counts during inference
- **Model Profiles**: Pre-configured layouts for Llama, Granite, Qwen, Phi, and Mistral
- **Layer Mapping**: Automatic mapping of observations to transformer layers
- **JSON Export**: Save capture data for analysis or visualization
- **REST API**: FastAPI backend for serving visualization data

---

## Installation

### Requirements

- Python 3.8+
- `llama-cpp-python` (with the cb_eval patch - see below)

### Install Spectra Vis

```bash
# Clone the repository
git clone https://github.com/3rdEyeVisuals/spectra-vis.git
cd spectra-vis

# Install dependencies
pip install -r backend/requirements.txt
```

---

## Patching llama-cpp-python

Spectra Vis requires a small patch to `llama-cpp-python` to expose the `cb_eval` callback. The callback mechanism already exists in llama.cpp - this patch simply exposes it to Python.

### The Patch (llama_cpp/llama.py)

Find the `Llama` class `__init__` method and add the `cb_eval` parameter:

```python
# In llama_cpp/llama.py, find the Llama class __init__ method

def __init__(
    self,
    model_path: str,
    # ... existing parameters ...
    cb_eval: Optional["ctypes.CFUNCTYPE"] = None,  # ADD THIS LINE
    cb_eval_user_data: Optional[ctypes.c_void_p] = None,  # ADD THIS LINE
    # ... rest of parameters ...
):
```

Then, in the same method where the context is created, add:

```python
# Find where llama_new_context_with_model is called
# Add these lines before the context creation:

if cb_eval is not None:
    self._ctx_params.cb_eval = cb_eval
    self._ctx_params.cb_eval_user_data = cb_eval_user_data
```

That's it! The `cb_eval` callback is already supported by the underlying llama.cpp library - this patch just exposes it to Python.

### Verify the Patch

```python
from llama_cpp import Llama
import inspect

sig = inspect.signature(Llama.__init__)
if 'cb_eval' in sig.parameters:
    print("Patch applied successfully!")
else:
    print("Patch not detected")
```

---

## Quick Start: Two-Step Workflow

Spectra Vis uses a simple two-step batch workflow for tensor discovery:

```
┌──────────────────────────┐         ┌──────────────────────────┐
│   1. CAPTURE TENSORS     │         │   2. VISUALIZE           │
│──────────────────────────│   ───►  │──────────────────────────│
│  Windows: Capture_Tensors.bat     │  Windows: start_server.bat│
│  macOS:   python capture_tensors.py│  macOS:   start_server.command
│  Linux:   python capture_tensors.py│  Linux:   ./start_server.sh│
└──────────────────────────┘         └──────────────────────────┘
           │                                    │
           ▼                                    ▼
      Interactive CLI                    Web Interface
      - Load any GGUF model              - 3D tensor grid
      - Run inference trials             - Heat map view
      - Auto-save to JSON                - Flow visualization
```

### Step 1: Capture Tensors

| Platform | Command |
|----------|---------|
| **Windows** | Double-click `Capture_Tensors.bat` |
| **macOS** | `python3 capture_tensors.py` |
| **Linux** | `python3 capture_tensors.py` |

The interactive tool will guide you through:

1. **Drag & drop your GGUF model** - Just paste the file path
2. **Select model family** - Llama, Granite, Qwen, Phi, or Mistral
3. **Enter prompts** - Add one or more prompts to test (or type `default` for samples)
4. **Watch the capture** - See real-time tensor observation counts
5. **Auto-save** - Results saved to `data/capture_<model>_<timestamp>.json`

```
============================================================
  SPECTRA VIS - Tensor Capture Tool
============================================================

Enter the path to your GGUF model file:
(You can drag & drop the file here)

> C:\models\llama-3.2-1b.Q4_K_M.gguf

Select your model family:
  1. Meta Llama (Llama 2, Llama 3, etc.)
  2. IBM Granite
  3. Alibaba Qwen
  4. Microsoft Phi
  5. Mistral / Mixtral

> 1

Prompt 1: What is the meaning of life?
Prompt 2: [Enter to finish]

Running inference trials...
[Trial 1/1] Tensors: 847 unique | Time: 1234ms

Saved to: data/capture_llama-3.2-1b.Q4_K_M_20250101_120000.json
```

### Step 2: Visualize Results

| Platform | Command |
|----------|---------|
| **Windows** | Double-click `start_server.bat` |
| **macOS** | Double-click `start_server.command` (or run `./start_server.sh`) |
| **Linux** | `chmod +x start_server.sh && ./start_server.sh` (first time only needs chmod) |

This launches both the backend API and frontend UI:
- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

Load your capture file in the web interface to explore:
- **3D Tensor Grid** - Layers × tensor types visualization
- **Heat Maps** - Activation intensity across the model
- **Flow Diagrams** - Data path through transformer blocks

---

## Advanced: Python API

For programmatic control, use the Python API directly:

```python
from spectra_vis import TensorCollector, create_callback
from llama_cpp import Llama

# Create collector
collector = TensorCollector(model_family="llama", verbose=True)
callback = create_callback(collector)

# Load model with callback (requires patched llama-cpp-python)
model = Llama(
    model_path="path/to/model.gguf",
    n_ctx=2048,
    cb_eval=callback
)

# Run inference with collection
collector.start_trial("What is the capital of France?")
response = model("What is the capital of France?", max_tokens=50)
trial = collector.end_trial(response["choices"][0]["text"])

# Save results
collector.save_to_json("capture_data.json")

print(f"Captured {len(trial.tensor_counts)} unique tensors")
```

### REST API

```bash
# Load a capture file
curl -X POST http://localhost:8000/api/load \
  -H "Content-Type: application/json" \
  -d '{"filepath": "capture_data.json"}'

# Analyze with model profile
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_family": "llama", "model_size": "7b"}'
```

---

## Supported Models

| Family  | Models                                    | Provider   |
|---------|-------------------------------------------|------------|
| Llama   | Llama, Llama 2, Llama 3, Llama 3.1/3.2   | Meta       |
| Granite | Granite 3b, 8b, 20b, Granite 3           | IBM        |
| Qwen    | Qwen, Qwen 2, Qwen 2.5                   | Alibaba    |
| Phi     | Phi-2, Phi-3 (mini, small, medium)       | Microsoft  |
| Mistral | Mistral 7B, Mixtral 8x7B, 8x22B          | Mistral AI |

---

## API Reference

### Endpoints

| Endpoint           | Method | Description                              |
|--------------------|--------|------------------------------------------|
| `/api/models`      | GET    | List supported model families            |
| `/api/files`       | GET    | List capture files in data directory     |
| `/api/load`        | POST   | Load a capture file                      |
| `/api/status`      | GET    | Get server status and loaded data info   |
| `/api/analyze`     | POST   | Analyze loaded data with model profile   |
| `/api/tensor-grid` | POST   | Get data formatted for 3D visualization  |
| `/api/colors`      | GET    | Get color scheme definitions             |
| `/api/flows`       | GET    | Get tensor flow definitions              |

---

## How It Works

1. **Callback Registration**: A ctypes callback function is registered with llama.cpp's `cb_eval` mechanism

2. **Tensor Observation**: During inference, llama.cpp calls the callback for each tensor being evaluated. The callback records:
   - Tensor memory address
   - Observation order (sequence number)
   - Observation count (how many times each tensor is accessed)

3. **Profile Mapping**: Using pre-defined model profiles, observations are mapped to:
   - Layer index (0 to N-1 for transformer layers)
   - Tensor type (attention Q/K/V, FFN, normalization, etc.)
   - Category (attention, feedforward, output)

4. **Visualization**: The mapped data can be rendered as:
   - 3D tensor grid (layers x tensor types)
   - Heat map showing activation intensity
   - Flow diagram showing data path through model

### Layer Convention

- **Layer -1**: Embedding (token_embd)
- **Layers 0 to N-1**: Transformer blocks
- **Layer 998**: Final normalization (output_norm)
- **Layer 999**: Output projection (logits)

---

## Project Structure

```
spectra-vis/
│
├── Capture_Tensors.bat         # ⭐ STEP 1: Interactive tensor capture (Windows)
├── capture_tensors.py          #    Cross-platform capture script
│
├── start_server.bat            # ⭐ STEP 2: Launch visualization (Windows)
├── start_server.sh             #    Linux/macOS launcher
├── start_server.command        #    macOS double-click launcher
│
├── spectra_vis/                # Core Python library
│   ├── __init__.py            #    Package exports
│   ├── collector.py           #    TensorCollector class
│   └── model_profiles.py      #    Model architecture profiles
│
├── backend/                    # FastAPI backend
│   ├── server.py              #    REST API server
│   └── requirements.txt       #    Python dependencies
│
├── frontend/                   # React + Three.js frontend
│   ├── src/                   #    React components
│   └── package.json           #    Node dependencies
│
├── data/                       # Capture output directory
│   └── capture_*.json         #    Saved tensor captures
│
├── examples/                   # Example scripts
│   ├── basic_capture.py       #    Simple capture example
│   └── analyze_results.py     #    Analysis example
│
├── docs/
│   └── TENSOR_GUIDE.md        # Tensor education guide
│
├── LICENSE                     # Non-commercial license
└── README.md                   # This file
```

---

## License

Spectra Vis is released under a custom non-commercial license.

**Permitted:**
- Personal use
- Educational use
- Research use
- Modification for personal projects

**Requires Permission:**
- Commercial use
- Redistribution in commercial products

**Required:**
- Credit to "3rdEyeVisuals" in any derivatives

See [LICENSE](LICENSE) for full terms.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

---

## Acknowledgments

- The llama.cpp team for the excellent inference library
- The open-source LLM community

---

*Copyright (c) 2025 3rdEyeVisuals*
