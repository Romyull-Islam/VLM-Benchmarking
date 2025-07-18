# Custom VLM Benchmark

This repository provides a customizable Python module to benchmark Visual Language Models (VLMs) on popular multimodal datasets.  
It builds on and extends the excellent [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) by OpenCompass, adapting it for lightweight usage, energy-aware setups, and additional flexibility.

## 🔗 Acknowledgment

This project is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which provides comprehensive support for benchmarking vision-language models.  
We thank the original authors for their open-source contribution, and this repository reuses and modifies parts of that framework for our custom experiments.

---

## 📦 Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for large models)(for Raspberry pi or cpu only device not mandatory)
- [git](https://git-scm.com/) (to clone VLMEvalKit)

---

## 🔧 Installation

1. **Clone this repository and VLMEvalKit:**

    ```bash
    git clone https://github.com/Romyull-Islam/VLM-Benchmarking.git
    git clone https://github.com/open-compass/VLMEvalKit.git  # if not already cloned
    ```

2. **Install dependencies:**

    It's recommended to use a virtual environment (conda or venv):

    ```bash
    pip install -r requirements.txt
    ```

    This will install:
    - PyTorch
    - Huggingface Transformers
    - pandas
    - tqdm
    - Pillow  
    and other dependencies used by VLMEvalKit and this script.

3. **(Optional) Set up CUDA:**

    For best performance:

    ```bash
    nvcc -V  # Verify CUDA version
    ```

    Ensure CUDA 11.7+ and appropriate NVIDIA drivers are installed.

4. **Huggingface token:**

    Create a file named `hf_token.txt` and paste your token inside it (used for gated models).

---

## 🚀 Usage

Run the main script:

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
