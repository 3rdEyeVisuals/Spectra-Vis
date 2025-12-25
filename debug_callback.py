"""
Debug script to test cb_eval callback registration.
"""

import sys
import ctypes
from pathlib import Path

print("=" * 60)
print("SPECTRA VIS - Callback Debug Test")
print("=" * 60)

# Test 1: Check llama_cpp imports
print("\n[TEST 1] Checking llama_cpp imports...")
try:
    from llama_cpp import Llama
    print("  OK: llama_cpp.Llama imported")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

try:
    from llama_cpp import llama_cpp
    print("  OK: llama_cpp.llama_cpp imported")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  Your llama-cpp-python may not expose the low-level module")

# Test 2: Check for callback type
print("\n[TEST 2] Checking for ggml_backend_sched_eval_callback...")
try:
    from llama_cpp import llama_cpp
    callback_type = llama_cpp.ggml_backend_sched_eval_callback
    print(f"  OK: Found callback type: {callback_type}")
except AttributeError as e:
    print(f"  FAIL: ggml_backend_sched_eval_callback not found")
    print(f"  Error: {e}")
    print("\n  Checking what IS available in llama_cpp module...")
    try:
        attrs = [a for a in dir(llama_cpp) if 'callback' in a.lower() or 'eval' in a.lower()]
        print(f"  Available callback/eval attrs: {attrs}")
    except:
        pass

# Test 3: Check Llama constructor signature
print("\n[TEST 3] Checking Llama constructor for cb_eval...")
import inspect
try:
    sig = inspect.signature(Llama.__init__)
    params = list(sig.parameters.keys())

    if 'cb_eval' in params:
        print("  OK: cb_eval parameter exists in Llama.__init__")
    else:
        print("  FAIL: cb_eval NOT in Llama.__init__ parameters")
        print(f"  Available params: {params}")

    if 'cb_eval_user_data' in params:
        print("  OK: cb_eval_user_data parameter exists")
    else:
        print("  WARN: cb_eval_user_data NOT in parameters")

except Exception as e:
    print(f"  Error inspecting: {e}")

# Test 4: Create a simple callback and test
print("\n[TEST 4] Creating test callback...")

callback_fired = {"count": 0}

try:
    from llama_cpp import llama_cpp

    @llama_cpp.ggml_backend_sched_eval_callback
    def test_callback(tensor_ptr, is_ask, user_data):
        callback_fired["count"] += 1
        if callback_fired["count"] <= 5:
            print(f"    [CALLBACK] tensor={tensor_ptr}, is_ask={is_ask}")
        return True

    print("  OK: Callback created with decorator")

except Exception as e:
    print(f"  FAIL: Could not create callback with decorator: {e}")
    print("  Trying ctypes fallback...")

    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_void_p
    )

    @callback_type
    def test_callback(tensor_ptr, is_ask, user_data):
        callback_fired["count"] += 1
        if callback_fired["count"] <= 5:
            print(f"    [CALLBACK] tensor={tensor_ptr}, is_ask={is_ask}")
        return True

    print("  OK: Callback created with ctypes fallback")

# Test 5: Load model and run inference
print("\n[TEST 5] Loading model with callback...")
print("  Enter path to a small GGUF model (or press Enter to skip):")
model_path = input("  > ").strip().strip('"').strip("'")

if model_path and Path(model_path).exists():
    try:
        print(f"\n  Loading {Path(model_path).name}...")
        model = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=-1,
            cb_eval=test_callback,
            cb_eval_user_data=None,
            verbose=False,
        )
        print("  OK: Model loaded with cb_eval")

        print("\n  Running inference...")
        response = model("Hello", max_tokens=10)

        print(f"\n  Response: {response['choices'][0]['text'][:50]}")
        print(f"  Callback fired: {callback_fired['count']} times")

        if callback_fired["count"] == 0:
            print("\n  WARNING: Callback never fired!")
            print("  This means cb_eval is accepted but not being called.")
            print("  Possible causes:")
            print("    1. cb_eval patch not applied to llama.cpp backend")
            print("    2. Callback type mismatch")
            print("    3. GPU backend bypassing callback")

    except TypeError as e:
        if "cb_eval" in str(e):
            print(f"  FAIL: cb_eval not accepted: {e}")
        else:
            print(f"  FAIL: {e}")
    except Exception as e:
        print(f"  FAIL: {e}")
else:
    print("  Skipped model test")

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)
input("\nPress Enter to exit...")
