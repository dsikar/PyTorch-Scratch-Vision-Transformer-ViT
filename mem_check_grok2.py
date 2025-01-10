import torch

def check_gpu_memory():
    if torch.cuda.is_available():
        # Get the current device (GPU)
        device = torch.cuda.current_device()
        
        # Get the name of the GPU
        gpu_name = torch.cuda.get_device_name(device)
        
        # Get memory usage
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Calculate available memory
        available_memory = total_memory - reserved_memory
        
        # Convert bytes to GB for readability
        total_memory_gb = total_memory / (1024 ** 3)
        available_memory_gb = available_memory / (1024 ** 3)
        allocated_memory_gb = allocated_memory / (1024 ** 3)
        
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory_gb:.2f} GB")
        print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
        print(f"Available Memory: {available_memory_gb:.2f} GB")
    else:
        print("No CUDA-capable device is detected")

# Run the function to check GPU memory
check_gpu_memory()
