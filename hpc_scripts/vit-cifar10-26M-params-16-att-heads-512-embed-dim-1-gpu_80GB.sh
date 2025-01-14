#!/bin/bash
#SBATCH -D /users/aczd097/git/vit/PyTorch-Scratch-Vision-Transformer-ViT # working directory
#SBATCH --job-name 26M16HCF                      # Job name
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk         # Where to send mail
#SBATCH --partition=preemptgpu			  # Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=2                          # Use 4 cores, most of the procesing happens on the GPU        
#SBATCH --mem=4GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                          # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100_80g:2					#will run on a 40gb card
#SBATCH -e outputs/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o outputs/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.

# from cmarshall's msg
# gres=gpu:a100:1 #will run on a 40gb card
# gres=gpu:a100_80g:1 #will run on a 80gb card

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

python main.py --dataset cifar10 --n_channels 3 --embed_dim 512 --n_attention_heads 16 --n_layers 12--image_size 32 --augmentation randaugment --epochs 5000

end=$(date +%s) # Record the end time in seconds since epoch
diff=$((end-start)) 

# Convert seconds to hours, minutes, and seconds
hours=$((diff / 3600))
minutes=$(( (diff % 3600) / 60 ))
seconds=$((diff % 60))

echo "python cifar10 vit 26M params, 512 embed dim, 12 layers - HPC gpu instance 80gb card instance script execution time: $hours hours, $minutes minutes, $seconds seconds"

