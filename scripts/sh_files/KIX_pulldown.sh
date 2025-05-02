#!/bin/bash
#SBATCH --job-name=KIX_alphapulldown      # Name the job for CREBBP Full
#SBATCH --partition=gpu                        # Specify the GPU partition
#SBATCH --mem=200G                             # Request memory (adjust if necessary)
#SBATCH --cpus-per-task=16                     # CPU cores (adjust if necessary)
#SBATCH --gres=lscratch:50,gpu:a100:1          # Request GPU and scratch space
#SBATCH --time=3-24:00:00                         # Job time limit (6 hours)
#SBATCH --output=output/KIX.log    # Unique output log for each job
#SBATCH --error=error/KIX.log      # Unique error log for each job
#SBATCH --mail-type=END,FAIL                   # Email on job completion or failure
#SBATCH --mail-user= # Replace with your email

set -e
set -x  # Enable debugging

# Load necessary modules
module load colabfold alphapulldown/0.30.7

# Change to the input directory
cd "/data/CBLCCBR/crebbp_ep300/crebbp/KIX"

# Run the AlphaPulldown job in this folder
run_multimer_jobs.py     --mode=pulldown     --num_cycle=3     --num_predictions_per_model=2     --output_path=pulldown_models     --protein_lists=bait.txt,candidate.txt     --monomer_objects_dir=pulldown_cf_msas

run_get_good_pae.sh --output_dir pulldown_models --cutoff=50

cd "/data/CBLCCBR/scripts/heterodimer"

source myconda

conda activate myenv

python3 creating_excel.py -output_dir=/data/CBLCCBR/crebbp_ep300/crebbp/KIX/pulldown_models

rm -rf /data/CBLCCBR/crebbp_ep300/crebbp/KIX/pulldown_cf_msas

