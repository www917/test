# Fix: qwen3_5 Model Type Not Recognized

## Problem

When loading a Qwen3.5 model checkpoint (e.g. `Qwen/Qwen3.5-4B`) with vLLM or transformers, you get:

```
KeyError: 'qwen3_5'

ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

## Root Cause

Qwen3.5 model support (`qwen3_5` model type) was added to the Hugging Face
`transformers` library in **version 5.2.0** (released February 16, 2026).

If your installed `transformers` version is older than 5.2.0, the `CONFIG_MAPPING`
in `configuration_auto.py` does not include the `qwen3_5` key, causing the
`KeyError`.

## Solution

### Option 1: Upgrade transformers (recommended)

```bash
pip install "transformers>=5.2.0"
```

### Option 2: Install transformers from source (if no release supports it yet)

```bash
pip install git+https://github.com/huggingface/transformers.git
```

### Option 3: Use the provided fix script

```bash
# Default conda env "show"
./fix_qwen3_5_model_recognition.sh

# Or specify a different conda env
./fix_qwen3_5_model_recognition.sh myenv
```

## Verification

After upgrading, verify the fix:

```python
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert 'qwen3_5' in CONFIG_MAPPING, "qwen3_5 not found"
print("qwen3_5 is recognized!")
```

## vLLM Compatibility

If you're using vLLM, ensure your vLLM version also supports Qwen3.5.
Check the [vLLM Qwen3.5 documentation](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_5/)
for version requirements.
