#!/bin/bash
# Convert saved HuggingFace models to GGUF format for Ollama
# Usage: bash scripts/convert_to_gguf.sh /workspace/results/run_XXXXXXXX

set -e

RESULTS_DIR=${1:?Usage: $0 <results_dir>}

echo "=== Installing llama.cpp conversion tools ==="
if [ ! -d "/workspace/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
    pip install -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt 2>/dev/null || true
fi

for METHOD in gene-conv lora; do
    MODEL_DIR="$RESULTS_DIR/$METHOD/model"
    if [ -d "$MODEL_DIR" ]; then
        echo ""
        echo "=== Converting $METHOD model to GGUF ==="
        GGUF_PATH="$RESULTS_DIR/$METHOD/model.gguf"

        python /workspace/llama.cpp/convert_hf_to_gguf.py \
            "$MODEL_DIR" \
            --outfile "$GGUF_PATH" \
            --outtype f16

        echo "GGUF saved: $GGUF_PATH"
        echo "Size: $(du -h "$GGUF_PATH" | cut -f1)"

        # Update Modelfile to point to GGUF
        cat > "$RESULTS_DIR/$METHOD/Modelfile.gguf" << EOF
FROM $GGUF_PATH

TEMPLATE """{{- if .System }}
<s>[INST] <<SYS>>
{{ .System }}
<</SYS>>
{{- end }}

{{ .Prompt }} [/INST] {{ .Response }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
EOF
        echo "Ollama Modelfile: $RESULTS_DIR/$METHOD/Modelfile.gguf"
    fi
done

echo ""
echo "=== Done ==="
echo "To import into Ollama:"
echo "  ollama create molly-gc -f $RESULTS_DIR/gene-conv/Modelfile.gguf"
echo "  ollama create molly-lora -f $RESULTS_DIR/lora/Modelfile.gguf"
