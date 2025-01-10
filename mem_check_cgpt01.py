import subprocess

def check_gpu_memory_smi():
    """
    Use nvidia-smi to get GPU memory usage (if torch.cuda.mem_get_info is unavailable).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip().split("\n")
        
        for idx, line in enumerate(output):
            free_mem, total_mem = line.split(", ")
            print(f"GPU {idx}: Free = {free_mem} MB, Total = {total_mem} MB")
            
    except FileNotFoundError:
        print("nvidia-smi not found. Please make sure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run nvidia-smi: {e}")

if __name__ == "__main__":
    check_gpu_memory_smi()

