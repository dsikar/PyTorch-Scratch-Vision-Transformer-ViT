#!/bin/bash
#SBATCH -D /users/aczd097/git/vit/PyTorch-Scratch-Vision-Transformer-ViT # working directory
#SBATCH --job-name C10_2GPU                      # Job name
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk         # Where to send mail
#SBATCH --partition=preemptgpu			  # Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU        
#SBATCH --mem=47GB                                # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                          # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100_80g:2					#will run on a 40gb card
#SBATCH -e outputs/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o outputs/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.

# from cmarshall's msg
## gres=gpu:a100:1 #will run on a 40gb card
## gres=gpu:a100_80g:2 #will run on a 80gb card

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

#Modules required
#module load python/3.7.12 # now loading through pipenv
module add gnu
#Run script
start=$(date +%s) # Record the start time in seconds since epoch

nvidia-smi
# or
python -c "import torch; print(torch.cuda.device_count()); [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

whoami

CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset cifar10 --n_channels 3 --embed_dim 1024 \
--n_attention_heads 16 --n_layers 12 --image_size 32 \
--patch_size 8 \
--batch_size 2048 \
--augmentation randaugment --checkpoint_frequency 100 --epochs 5000

end=$(date +%s) # Record the end time in seconds since epoch
diff=$((end-start)) 

# Convert seconds to hours, minutes, and seconds
hours=$((diff / 3600))
minutes=$(( (diff % 3600) / 60 ))
seconds=$((diff % 60))

echo "python cifar10 vit 101M params, batch size 1024, patch size 8, random augmentation - HPC gpu instance 80gb card instance script execution time: $hours hours, $minutes minutes, $seconds seconds"

