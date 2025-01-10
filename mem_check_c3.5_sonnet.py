import torch

def check_gpu_memory():
    """
    Check and display GPU memory usage for all available GPU devices.
    Returns a dictionary with memory info for each GPU.
    """
    if not torch.cuda.is_available():
        return "No GPU available. Please check if CUDA is installed properly."
    
    memory_info = {}
    
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(device)
        
        # Get memory information in bytes, convert to GB
        total_memory = props.total_memory / 1024**3  # Convert to GB
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3
        free_memory = total_memory - allocated_memory
        
        memory_info[f'gpu_{i}'] = {
            'device_name': props.name,
            'total_memory_GB': round(total_memory, 2),
            'allocated_memory_GB': round(allocated_memory, 2),
            'reserved_memory_GB': round(reserved_memory, 2),
            'free_memory_GB': round(free_memory, 2)
        }
    
    return memory_info

# Example usage
if __name__ == "__main__":
    memory_stats = check_gpu_memory()
    
    if isinstance(memory_stats, str):
        print(memory_stats)
    else:
        for gpu, stats in memory_stats.items():
            print(f"\n{gpu.upper()} ({stats['device_name']}):")
            print(f"Total Memory: {stats['total_memory_GB']:.2f} GB")
            print(f"Allocated Memory: {stats['allocated_memory_GB']:.2f} GB")
            print(f"Reserved Memory: {stats['reserved_memory_GB']:.2f} GB")
            print(f"Free Memory: {stats['free_memory_GB']:.2f} GB")
