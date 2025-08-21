# Medical Report Parser with vLLM

A configurable pipeline for extracting structured data from medical reports using LLMs accessible on HuggingFace.

## Key Features

- **Structured Information Extraction**: Converts unstructured medical reports into structured data.
- **Customizable Extraction**: Define fields and validation rules via YAML.
- **Multiple LLMs and Prompting strategies**: Define LLM using model parameter files, and experiment with different prompting strategies.
- **Performance calculation**: If providing ground truth, metrics can be automatically be calculated and visualized.

---

## Installation

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Clone repository
git clone https://github.com/Douwe-Spaanderman/MedicalRecordLLM
cd MedicalRecordLLM

# Install dependencies
pip install -r requirements.txt
```
---

## How to Use

The parser can be executed using the provided `run.py` script. Below is an example usage:

```bash
python run.py \
  --input path/to/input.csv \
  --output path/to/output.csv \
  --format csv \
  --prompt-method ZeroShot \
  --prompt-config configs/prompt.yaml \
  --params-config configs/model.yaml \
  --gpus 4
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input` | **Required.** Path to input file (CSV or JSON). |
| `-o, --output` | **Required.** Path to save the processed output file. |
| `-f, --format` | **Required.** Input file format (`csv` or `json`). |
| `-pm, --prompt-method` | **Required.** Prompting method (`ZeroShot`, `OneShot`, `FewShot`, `CoT`, `SelfConsistency`, `PromptGraph`). |
| `-pc, --prompt-config` | **Required.** Path to YAML config file for prompt definitions. |
| `-pa, --params-config` | Path to YAML config for model parameters (default: `config_parameters.yaml`). |
| `-u, --base-url` | Base URL for the LLM API (default: `http://localhost:8000/v1`). |
| `--api-key` | API key for the LLM service (default: `DummyAPIKey`). |
| `--text-key` | Key containing text in a JSON input (default: `'Text'`). |
| `--report-type-key` | Key containing report type in a JSON input (default: `'reportType'`). |
| `--text-col` | Text column name in a CSV input (default: `'Text'`). |
| `--patient-id-col` | Patient ID column name in a CSV input (default: `'patientID'`). |
| `-b, --batch-size` | Internal batch size for processing reports (default: `None`). |
| `-t, --timeout` | Timeout for each LLM request in seconds (default: `60`). |
| `-mc, --max-concurrent` | Maximum number of concurrent requests (default: `32`). |
| `-se, --select_example` | 1-based index of the example to use for example-based prompts (default: `None`). |
| `-r, --regex` | Path to a JSON file with regex patterns for pre-extraction. |
| `--save-raw` | Save the raw model output to a file. |
| `-v, --verbose` | Print intermediate outputs for debugging (slows execution). |
| `--dry-run` | Test the workflow without calling the LLM. |

---

## Defining Extraction Rules with YAML

The pipeline is configured via YAML files that defines the model parameters, and the prompt, including which fields to extract and how to validate them, and optionally an example case. In resources you can find multiple example for both [Prompt YAML](resources/prompt_configs/) and [Parameter YAML](resources/model_configs/).

## Configuration Files

### 1. Model Parameters (`config_parameters.yaml`)

```yaml
model: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" # Full path to HuggingFace
max_model_len: 64000
max_tokens: 4096
temperature: 0.6
top_p: 0.95
top_k: 50
frequency_penalty: 0.0
presence_penalty: 0.0
repetition_penalty: 1.2

self_consistency_sampling:
  num_samples: 3
  temperature: [0.5, 0.6, 0.7]
```

### 2. Prompt Configuration (`prompt_config.yaml`)

```yaml
field_instructions:
  - name: "diagnosis"
    type: "string"
    required: true
    constraints: "Primary diagnosis from pathology report"

task: |
  [TASK] Extract information into JSON structure...

examples:
  - input: |
      anonymized for github
  - reasoning: |
      some reasong to select output Y
  - output: |
      {"diagnosis": "Y"}
```

## Optional: Running a full experiment

Alternatively, when experimenting with multiple LLMs and Prompting strategies, you can automatically run all (including serving with vLLM) using this [single script](experiments/run_use_case.py). Example usage below runs all "medium" models, with all available prompting strategies on 4 GPUs, and initializes the vLLM server all at once:

```bash
python experiments/run_use_case.py \
  --data-path path/to/input.csv \
  --output-dir path/to/output_dir/ \
  --prompt-config configs/prompt.yaml \
  --format csv \
  --model-configs "model_configs/NVIDIA-Nemotron‑Super‑49B.yaml" "/model_configs/Qwen2.5-72B-Instruct.yaml" "model_configs/Llama-4-Scout.yaml" "model_configs/Llama-3-OpenBioLLM-70B.yaml" "model_configs/Llama-3-Med42-70B.yaml" \
  --gpu-parallelization 4 \
  --node-parallelization 1 \
  --overrides-config model_configs/overrides_config.yaml \
  --vllm-server \
  --with-balanced-accuracy
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--data-path` | **Required.** Path to input CSV or JSON file. |
| `--output-dir` | **Required.** Directory to save output files. |
| `--prompt-config` | **Required.** Path to YAML file with prompt templates. |
| `--model-configs` | **Required.** List of YAML model config files (space-separated). |
| `--prompt-methods` | Prompting methods to try (default: all available methods). |
| `--format` | **Required.** Input file format (`csv` or `json`). |
| `--patient-id-col` | Patient ID column name (default: `Patient-ID`). |
| `--timeout` | Timeout for each request in seconds (default: `240`). |
| `--max-concurrent` | Maximum concurrent requests (default: `64`). |
| `--overrides-config` | YAML file with timeout/concurrent overrides for models and methods. Usefull when trying multiple different LLMs with varying timeout and max-concurrent. |
| `--vllm-base-url` | Base URL for vLLM API (default: `http://localhost:8000/v1/`). |
| `--vllm-server` | Run vLLM server for each model configuration. |
| `--precision` | Data type for model weights and activations (default: `bfloat16`). |
| `--gpu-parallelization` | Number of GPUs to use for parallelization (default: `1`). |
| `--node-parallelization` | Number of nodes to use for parallelization (default: `1`). |
| `--vllm-timeout` | Timeout for vLLM operations (default: `600`). |
| `--with-balanced-accuracy` | Use balanced accuracy and macro average for performance calculation. |
| `--measurement-run` | Perform only measurement without prompting LLM. |
| `--dry-run` | Print commands without running them. |