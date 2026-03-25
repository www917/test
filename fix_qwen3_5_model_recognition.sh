#!/bin/bash
# Fix for: KeyError: 'qwen3_5'
# The checkpoint you are trying to load has model type `qwen3_5`
# but Transformers does not recognize this architecture.
#
# Root cause: Qwen3.5 support was added in transformers v5.2.0 (Feb 16, 2026).
# Your installed version is older and lacks the qwen3_5 model type registration
# in CONFIG_MAPPING (configuration_auto.py).
#
# This script upgrades transformers (and optionally vLLM) to versions
# that include Qwen3.5 support.

set -e

CONDA_ENV="${1:-show}"

echo "=== Fixing qwen3_5 model recognition ==="
echo "Target conda environment: $CONDA_ENV"
echo ""

# Activate conda env if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Check current transformers version
echo ""
echo "Current transformers version:"
python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "  (not installed)"

echo ""
echo "Current vLLM version:"
python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "  (not installed)"

# Upgrade transformers to a version that supports qwen3_5 (>= 5.2.0)
echo ""
echo "=== Upgrading transformers to >= 5.2.0 (Qwen3.5 support) ==="
pip install "transformers>=5.2.0"

# Verify the fix
echo ""
echo "=== Verifying qwen3_5 is now recognized ==="
python -c "
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if 'qwen3_5' in CONFIG_MAPPING:
    print('SUCCESS: qwen3_5 model type is now recognized')
else:
    print('WARNING: qwen3_5 still not found in CONFIG_MAPPING')
    print('You may need to install transformers from source:')
    print('  pip install git+https://github.com/huggingface/transformers.git')
"

echo ""
echo "Updated transformers version:"
python -c "import transformers; print(transformers.__version__)"

echo ""
echo "=== Done ==="
echo ""
echo "If you are also using vLLM, make sure your vLLM version supports Qwen3.5."
echo "Check: https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_5/"
