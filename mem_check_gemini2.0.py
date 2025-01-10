import torch

def get_gpu_memory_info():
    """
    Retrieves information about GPU memory usage.

    Returns:
        dict or None: A dictionary containing memory information (total, allocated, reserved, free) in GB,
                      or None if no CUDA-enabled GPU is available.
                      Returns -1 if there is an exception.
    """
    if not torch.cuda.is_available():
        print("No CUDA-enabled GPU found.")
        return None

    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Total memory in GB
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # Allocated memory in GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3) # Reserved memory in GB
        free_memory = total_memory - allocated_memory # Free memory in GB. More accurate than reserved memory.

        memory_info = {
            "total": total_memory,
            "allocated": allocated_memory,
            "reserved": reserved_memory,
            "free": free_memory
        }
        return memory_info

    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

if __name__ == "__main__":
    memory_data = get_gpu_memory_info()

    if memory_data is None:
        pass
    elif memory_data == -1:
        print("Error getting memory info")
    else:
        print("GPU Memory Information:")
        for key, value in memory_data.items():
            print(f"{key.capitalize()}: {value:.2f} GB")

        # Example of how to use the free memory:
        free_mem_gb = memory_data["free"]
        if free_mem_gb > 4:  # Example: Check if more than 4GB is free
            print("Sufficient free GPU memory available.")
        else:
            print("Insufficient free GPU memory.")

        # Another example of how to check free memory using reserved memory
        free_mem_gb_reserved = memory_data['total'] - memory_data['reserved']
        print(f"Free memory based on reserved memory: {free_mem_gb_reserved:.2f}")
        if free_mem_gb_reserved > 4:
            print("Sufficient free GPU memory available based on reserved memory.")
        else:
            print("Insufficient free GPU memory based on reserved memory.")
