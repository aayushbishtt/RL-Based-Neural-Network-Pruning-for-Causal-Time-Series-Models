#!/bin/bash
#SBATCH --job-name=rl_pruning_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00           # 12 hours max
#SBATCH --partition=gpu           # Request GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G                 # Request 32GB RAM

# Exit on error
set -e

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

# 1. Load the necessary modules on Ruche
module purge
module load anaconda3/2023.09-0/none-none
module load cuda/12.2.2/none-none

# 2. Activate your conda environment (assuming it's named 'rl_env')
source activate rl_env

# 3. Move to the project directory
# SLURM starts the job in the directory where sbatch was called
cd $SLURM_SUBMIT_DIR

# 4. Run the training pipeline
echo "Starting training..."

# You can change 'full_pipeline' to 'train_base' or 'train_agent' as needed
python main.py full_pipeline --epochs 100 --episodes 500

echo "Job finished at: $(date)"
