import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("\nNo CUDA available. To use GPU, you need to:")
        print("1. Have a CUDA-capable GPU")
        print("2. Install CUDA Toolkit")
        print("3. Install PyTorch with CUDA support")
        print("\nTo install PyTorch with CUDA support, use:")
        print("pip uninstall torch")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()