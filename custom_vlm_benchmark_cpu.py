import os
import torch
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
import warnings
from tabulate import tabulate
import time
from functools import partial


torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device('cpu'))

# Suppress Flash Attention warnings
warnings.filterwarnings("ignore", message="Flash Attention is not available")

# Import necessary components from VLMEvalKit
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.vlm.smolvlm import SmolVLM2
from vlmeval.vlm.moondream import Moondream2
#from vlmeval.vlm.llava.llava import 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RunBenchmarkCPU')

# Patch SmolVLM2 to use CPU
class CPUSmolVLM2(SmolVLM2):
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        assert os.path.exists(model_path) or len(model_path.split('/')) == 2

        self.sampling_frames = 64
        # Set resolution based on model
        if "SmolVLM2-2.2B" in model_path:
            self.resolution = 384
        elif "SmolVLM2-256M" in model_path or "SmolVLM2-500M" in model_path:
            self.resolution = 512
        else:
            raise ValueError(f"Unknown model {model_path}, cannot determine resolution")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to("cpu")  # Force CPU usage

        kwargs_default = {"max_new_tokens": 2048, "do_sample": False, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )

# Patch Moondream2 to use CPU
class CPUMoondream2(Moondream2):
    def __init__(self, model_path="vikhyatk/moondream2", revision="2025-01-09", **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            logging.critical(
                """Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
                please intall torchvision>=0.16."""
            )
            raise e

        assert os.path.exists(model_path) or len(model_path.split('/')) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",  # Force CPU usage
            revision=revision,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config.")
        
        
# Patch h2ovl-mississippi models to use CPU
class CPUH2OVLMississippi:
    def __init__(self, model_path="h2oai/h2ovl-mississippi-1b", **kwargs):
        from transformers import AutoProcessor, AutoModelForCausalLM

        assert os.path.exists(model_path) or len(model_path.split('/')) == 2

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

        kwargs_default = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        
        
        

class CustomVLMBenchmark:
    def __init__(self, work_dir: str = './outputs', force_cpu: bool = True, dataset_percentage: int = 100):
        """Initialize the benchmark with output directory.
        
        Args:
            work_dir (str): Directory to store outputs
            force_cpu (bool): Whether to force CPU execution (default True for CPU version)
            dataset_percentage (int): Percentage of dataset to use (5-100, default 100)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.force_cpu = force_cpu
        
        # Validate dataset percentage
        if not 5 <= dataset_percentage <= 100:
            raise ValueError("dataset_percentage must be between 5 and 100")
        self.dataset_percentage = dataset_percentage
        
        # Define supported models and datasets
        self.supported_models = {
            'InternVL2_5-4B-MPO': 'InternVL2_5-4B-MPO',
            'InternVL2_5-1B-MPO': 'InternVL2_5-1B-MPO',
            'InternVL2_5-1B': 'InternVL2_5-1B',
            'Moondream2': 'Moondream2',
            'SmolVLM2-256M': 'SmolVLM2-256M',
            'h2ovl-mississippi-1b': 'h2ovl-mississippi-1b',
        }
        
        self.supported_datasets = [
            'MMBench_DEV_EN',
            'SEEDBench_IMG',
            'MME',
            'MMStar',
            'MMVet'
        ]

    def run_benchmark(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Run benchmark for a specific model and dataset."""
        logger.info(f"Running benchmark for {model_name} on {dataset_name} using {self.dataset_percentage}% of the dataset")
        
        # Create output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.work_dir / f"{model_name}_{dataset_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Force CPU execution
            logger.info("Setting up CPU execution environment...")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["VLLM_TARGET_DEVICE"] = "cpu"
            torch.set_default_tensor_type(torch.FloatTensor)
            
            # Verify CPU is being used
            if torch.cuda.is_available():
                logger.warning("CUDA is still available despite environment settings!")
            else:
                logger.info("Successfully disabled CUDA - running on CPU")
            
            # Build model with CPU version for specific models
            logger.info(f"Building model: {model_name}")
            #if model_name == 'SmolVLM2-256M':
            #    model = CPUSmolVLM2(model_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
                
            if model_name == 'SmolVLM2-256M' or model_name == 'SmolVLM2-500M':
                model = CPUSmolVLM2(model_path=f"HuggingFaceTB/{model_name}-Video-Instruct")    
            elif model_name == 'Moondream2':
                model = CPUMoondream2(model_path="vikhyatk/moondream2")
            elif model_name == 'h2ovl-mississippi-1b':
                model = CPUH2OVLMississippi(model_path="h2oai/h2ovl-mississippi-1b")
            elif model_name == 'h2ovl-mississippi-800m':
                model = CPUH2OVLMississippi(model_path="h2oai/h2ovl-mississippi-800m")
            
            else:
                model = supported_VLM[model_name]()
                # Force model to CPU
                if hasattr(model, 'model'):
                    model.model = model.model.to('cpu')
                if hasattr(model, 'device'):
                    model.device = torch.device('cpu')
            
            # Build dataset
            logger.info(f"Building dataset: {dataset_name}")
            dataset = build_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Failed to build dataset: {dataset_name}")
            
            # Apply dataset subsetting if not using 100%
            if self.dataset_percentage < 100:
                logger.info(f"Using {self.dataset_percentage}% of the dataset")
                total_samples = len(dataset.data)
                subset_size = int(total_samples * (self.dataset_percentage / 100))
                # Randomly sample without replacement
                dataset.data = dataset.data.sample(n=subset_size, random_state=42)
                dataset.data = dataset.data.reset_index(drop=True)
                logger.info(f"Dataset reduced from {total_samples} to {subset_size} samples")
            
            # Run inference
            result_file = f"{model_name}_{dataset_name}.xlsx"
            result_path = output_dir / result_file
            
            # Run inference and wait for completion
            logger.info("Starting inference...")
            model = infer_data_job(
                model,
                work_dir=str(output_dir),
                model_name=model_name,
                dataset=dataset,
                verbose=True,
                api_nproc=4
            )
            
            # Wait for the result file to be created (with timeout)
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            while not result_path.exists():
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Timeout waiting for result file: {result_path}")
                time.sleep(1)
            
            # Evaluate results
            logger.info("Evaluating results...")
            eval_results = dataset.evaluate(str(result_path))
            
            # Save results
            if isinstance(eval_results, dict):
                scores_file = output_dir / f"scores_{model_name}_{dataset_name}.json"
                with open(scores_file, 'w') as f:
                    json.dump(eval_results, f, indent=4)
                
                # Print results in a formatted table
                logger.info("\nEvaluation Results:")
                df = pd.DataFrame.from_dict(eval_results, orient='index')
                logger.info("\n" + tabulate(df, headers='keys', tablefmt='psql'))
                
            elif isinstance(eval_results, pd.DataFrame):
                scores_file = output_dir / f"scores_{model_name}_{dataset_name}.xlsx"
                eval_results.to_excel(scores_file)
                
                logger.info("\nEvaluation Results:")
                logger.info("\n" + tabulate(eval_results, headers='keys', tablefmt='psql'))
            
            return {
                'model': model_name,
                'dataset': dataset_name,
                'dataset_percentage': self.dataset_percentage,
                'results': eval_results,
                'output_dir': str(output_dir),
                'predictions_file': str(result_path),
                'scores_file': str(scores_file)
            }
            
        except Exception as e:
            logger.error(f"Error running benchmark for {model_name} on {dataset_name}: {str(e)}")
            return {
                'model': model_name,
                'dataset': dataset_name,
                'dataset_percentage': self.dataset_percentage,
                'error': str(e)
            }

def main():
    """Main function to run the benchmark on CPU."""
    logger.info(f"PyTorch threads set to: {torch.get_num_threads()}")
    # Manually set model and dataset
    model_name = 'h2ovl-mississippi-800m'  # Choose from supported models
    dataset_name = 'MME'  # Choose from supported datasets
    dataset_percentage = 5  # Use 5% of the dataset
    
    logger.info(f"Starting benchmark for {model_name} on {dataset_name} on CPU using {dataset_percentage}% of the dataset...")
    
    # Initialize benchmark with dataset percentage
    benchmark = CustomVLMBenchmark(dataset_percentage=dataset_percentage)
    
    # Run benchmark
    result = benchmark.run_benchmark(model_name, dataset_name)
    
    # Print summary
    if 'error' in result:
        logger.error(f"Benchmark failed: {result['error']}")
    else:
        logger.info("\nBenchmark Summary:")
        logger.info(f"Model: {result['model']}")
        logger.info(f"Dataset: {result['dataset']}")
        logger.info(f"Dataset Percentage Used: {result['dataset_percentage']}%")
        logger.info(f"Output Directory: {result['output_dir']}")
        logger.info(f"Predictions File: {result['predictions_file']}")
        logger.info(f"Scores File: {result['scores_file']}")

if __name__ == "__main__":
    main()

 
