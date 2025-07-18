import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
from datetime import datetime
import logging
import warnings
from tabulate import tabulate
import time

# Suppress Flash Attention warnings
warnings.filterwarnings("ignore", message="Flash Attention is not available")

# Import necessary components from VLMEvalKit
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CustomVLMBenchmark')

device = "cuda" if torch.cuda.is_available() else "cpu" #for SmolVLM not needed

class CustomVLMBenchmark:
    def __init__(self, work_dir: str = './outputs', force_cpu: bool = False, dataset_percentage: int = 100):
        """Initialize the benchmark with output directory.
        
        Args:
            work_dir (str): Directory to store outputs
            force_cpu (bool): Whether to force CPU execution
            dataset_percentage (int): Percentage of dataset to use (5-100, default 100)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.device = device   #for SmolVLM not needed
        #self.force_cpu = force_cpu  ##for SmolVLM needed
        
        # Validate dataset percentage
        if not 5 <= dataset_percentage <= 100:
            raise ValueError("dataset_percentage must be between 5 and 100")
        self.dataset_percentage = dataset_percentage
        
        # Define supported models and datasets
        self.supported_models = {
            'InternVL2_5-4B-MPO': 'InternVL2_5-4B-MPO',
            'InternVL2_5-1B-MPO': 'InternVL2_5-1B-MPO',
            # 'SAIL-VL-1.5-2B': 'SAIL-VL-1.5-2B',
            #'h2ovl-mississippi-2b': 'h2ovl-mississippi-2b',
            'h2ovl-mississippi-1b': 'h2ovl-mississippi-1b', #only works with cuda
            #'llava_onevision_qwen2_0.5b_si': 'llava_onevision_qwen2_0.5b_si',
            'Moondream2': 'Moondream2',
            'SmolVLM2-256M': 'SmolVLM2-256M',
            'h2ovl-mississippi-800m': 'h2ovl-mississippi-800m',
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
            logger.info("Setting up GPU execution environment...") #for SmolVLM not needed
            os.environ["CUDA_VISIBLE_DEVICES"] = "" ##for SmolVLM not needed
            os.environ["VLLM_TARGET_DEVICE"] = "cuda" #for SmolVLM not needed
            torch.set_default_tensor_type(torch.FloatTensor) #for SmolVLM not needed
            # Build model
           
            #model = supported_VLM[self.supported_models[model_name]]() #for SmolVLM needed
            model = supported_VLM[model_name]() # works for moondream2


            # Optional: only move if appropriate
            if hasattr(model, 'model') and hasattr(model.model, 'to') and callable(model.model.to):
                model.model = model.model.to('cuda')

            if hasattr(model, 'device') and isinstance(model.device, torch.device):
                model.device = torch.device('cuda')

            #if hasattr(model, 'model'):   # works for moondream2
            #    model.model =model.model.to('cuda')
            #if hasattr(model, 'device'):   # works for moondream2
            #    model.device =model.device.to('cuda')

            
            
            # Build dataset
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

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks for all supported models and datasets."""
        all_results = {}
        
        for model_name in self.supported_models:
            model_results = {}
            for dataset_name in self.supported_datasets:
                result = self.run_benchmark(model_name, dataset_name)
                model_results[dataset_name] = result
            all_results[model_name] = model_results
            
            self._save_overall_results(model_name, model_results)
        
        return all_results

    def _save_overall_results(self, model_name: str, results: Dict[str, Any]):
        """Save overall results for a model across all datasets."""
        summary = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for dataset_name, result in results.items():
            if 'error' in result:
                summary['datasets'][dataset_name] = {'status': 'error', 'error': result['error']}
            else:
                summary['datasets'][dataset_name] = {
                    'status': 'success',
                    'output_dir': result['output_dir']
                }
                if isinstance(result['results'], dict):
                    summary['datasets'][dataset_name]['scores'] = result['results']
        
        output_file = self.work_dir / f"{model_name}_overall_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Saved overall results for {model_name} to {output_file}")

def main():
    """Main function to run the benchmark."""
    # Manually set model and dataset
    model_name = 'h2ovl-mississippi-800m'  # Choose from: 'InternVL2_5-4B-MPO', 'SmolVLM2-256M','SmolVLM2-500M' 'Moondream2'
    dataset_name = 'MME'  # Choose from: 'MMBench_DEV_EN', 'SEEDBench_IMG', 'MME', 'MMStar', 'MMVet'
    dataset_percentage = 5  # Use 5% of the dataset
    
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

