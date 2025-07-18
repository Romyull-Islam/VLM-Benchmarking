# Custom VLM Benchmark

This repository provides a Python module to benchmark Visual Language Models (VLMs) on popular multi-modal datasets using code adapted from VLMEvalKit.

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for large models)
- [git](https://git-scm.com/) (for cloning VLMEvalKit)

## Installation

1. **Clone the repository and VLMEvalKit:**

    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    # (If not already present in your workspace)
    ```

2. **Install dependencies:**

    It is recommended to use a virtual environment (e.g., conda or venv).

    ```bash
    pip install -r requirements.txt
    ```

    This will install PyTorch, Transformers, pandas, tqdm, Pillow, and other required packages. VLMEvalKit will be installed in editable mode.

3. **Huggingface token:**

    Put your huggingface token in `hf_token.txt` file.

4. **(Optional) Set up CUDA:**

    - For best performance, ensure you have CUDA 11.7+ and the appropriate NVIDIA drivers installed.
    - Check your CUDA version with:
      ```bash
      nvcc -V
      ```

## Usage

Run the benchmark script with your desired model and dataset:

```bash
python custom_vlm_benchmark.py
```

Edit the `main()` function in `custom_vlm_benchmark.py` to select:
- **Model:**
    - `InternVL2_5-4B-MPO`
    - `Moondream2`
    - `SmolVLM2-256M`
- **Dataset:**
    - `MMBench`
    - `SEEDBench_IMG`
    - `MMStar`
    - `MME`

The script will print results to the console and save them into './outputs' folder.

## Notes

- You can use either Huggingface Hub models or local checkpoints by editing the `model_path` in the script.
- Some models and datasets require significant GPU memory.
- VLMEvalKit will automatically download datasets and cache them locally.
- For more models and datasets, extend the `self.supported_models` and `self.supported_datasets` inside the CustomVLMBenchmark class of the script.

## Troubleshooting

- If you encounter CUDA or memory errors, ensure your GPU and drivers are compatible and have enough memory.
- If you see missing dependency errors, re-run `pip install -r requirements.txt`.
- For issues with VLMEvalKit, consult their [GitHub repository](https://github.com/open-compass/VLMEvalKit).
