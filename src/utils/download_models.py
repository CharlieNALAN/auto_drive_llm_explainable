#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import urllib.request
import subprocess
import platform
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# URLs for pre-trained models
MODEL_URLS = {
    # YOLOv8 models
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    
    # LLaMA model (these are large, so we use a smaller placeholder for demonstration)
    # In reality, you would download the full model from the official source
    'llama-2-7b-chat.gguf': None,  # Placeholder, will provide instructions
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_models(args):
    """Download all required models."""
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to {models_dir.absolute()}")
    
    # Download YOLO models
    if args.perception or args.all:
        print("\n=== Downloading perception models ===")
        yolo_model = 'yolov8n.pt' if args.small else 'yolov8s.pt'
        yolo_path = models_dir / yolo_model
        
        if not yolo_path.exists() or args.force:
            print(f"Downloading {yolo_model}...")
            download_url(MODEL_URLS[yolo_model], yolo_path)
        else:
            print(f"{yolo_model} already exists, skipping. Use --force to re-download.")
    
    # Download LLM models
    if args.llm or args.all:
        print("\n=== LLM model setup ===")
        llm_path = models_dir / 'llama-2-7b-chat.gguf'
        
        if not llm_path.exists() or args.force:
            print("For LLaMA models, due to their size and licensing restrictions, please:")
            print("1. Visit https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
            print("2. Download llama-2-7b-chat.Q4_K_M.gguf (or another variant)")
            print("3. Rename it to llama-2-7b-chat.gguf and place it in the models directory")
            print("\nAlternatively, you can use the OpenAI or Anthropic API by setting the appropriate environment variables:")
            print("- For OpenAI: export OPENAI_API_KEY=your_api_key")
            print("- For Anthropic: export ANTHROPIC_API_KEY=your_api_key")
            
            # Create a mock model file for testing if requested
            if args.mock:
                print("\nCreating mock LLM model file for testing...")
                with open(llm_path, 'w') as f:
                    f.write("This is a mock LLM model file for testing purposes.")
                print(f"Mock model created at {llm_path}")
        else:
            print(f"LLM model already exists at {llm_path}, skipping. Use --force to recreate.")
    
    # Install required Python packages
    if args.install_deps:
        print("\n=== Installing Python dependencies ===")
        
        # Basic dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Extra dependencies based on selected options
        if args.llm or args.all:
            print("Installing LLM dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            
            if args.mock:
                subprocess.run([sys.executable, "-m", "pip", "install", "openai", "anthropic"])
    
    print("\n=== Download complete! ===")
    print(f"Models are stored in {models_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for the autonomous driving system")
    parser.add_argument('--output-dir', default='models', help='Directory to store the downloaded models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--perception', action='store_true', help='Download perception models (YOLO)')
    parser.add_argument('--llm', action='store_true', help='Set up LLM (instructions for downloading)')
    parser.add_argument('--small', action='store_true', help='Download smaller models where available')
    parser.add_argument('--force', action='store_true', help='Force re-download of existing models')
    parser.add_argument('--mock', action='store_true', help='Create mock model files for testing')
    parser.add_argument('--install-deps', action='store_true', help='Install required Python dependencies')
    
    args = parser.parse_args()
    
    # If no specific models are selected, download all
    if not (args.all or args.perception or args.llm):
        args.all = True
    
    download_models(args)

if __name__ == "__main__":
    main() 