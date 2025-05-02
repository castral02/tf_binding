import os

# Set the parent directory containing the folders
PARENT_DIR = "/data/CBLCCBR/crebbp_ep300/crebbp"  # Change this to your actual directory path
SCRIPT_DIR = os.path.join(PARENT_DIR, "scripts/generated_pulldown_scripts")

# Create the directory for generated scripts if it doesn't exist
os.makedirs(SCRIPT_DIR, exist_ok=True)

# Loop through each folder in the directory
for folder_name in os.listdir(PARENT_DIR):
    folder_path = os.path.join(PARENT_DIR, folder_name)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        script_path = os.path.join(SCRIPT_DIR, f"{folder_name}.sh")

        # Write the script content
        with open(script_path, "w") as script_file:
            script_file.write(f"""#!/bin/bash
#SBATCH --job-name={folder_name}_alphapulldown      # Name the job for CREBBP Full
#SBATCH --partition=gpu                        # Specify the GPU partition
#SBATCH --mem=200G                             # Request memory (adjust if necessary)
#SBATCH --cpus-per-task=16                     # CPU cores (adjust if necessary)
#SBATCH --gres=lscratch:50,gpu:a100:1          # Request GPU and scratch space
#SBATCH --time=3-24:00:00                         # Job time limit (6 hours)
#SBATCH --output=output/{folder_name}.log    # Unique output log for each job
#SBATCH --error=error/{folder_name}.log      # Unique error log for each job
#SBATCH --mail-type=END,FAIL                   # Email on job completion or failure
#SBATCH --mail-user=   # Replace with your email

set -e
set -x  # Enable debugging

# Load necessary modules
module load colabfold alphapulldown

# Change to the input directory
cd "{folder_path}"

# Run the AlphaPulldown job in this folder
run_multimer_jobs.py \
    --mode=pulldown \
    --num_cycle=3 \
    --num_predictions_per_model=2 \
    --output_path=pulldown_models \
    --protein_lists=bait.txt,candidate.txt \
    --monomer_objects_dir=pulldown_cf_msas

run_get_good_pae.sh --output_dir pulldown_models --cutoff=50

cd "/data/CBLCCBR/scripts/heterodimer"

source myconda

conda activate myenv

python3 creating_excel.py -output_dir={folder_path}/pulldown_models

rm -rf {folder_path}/pulldown_cf_msas

""")

        # Make the script executable
        os.chmod(script_path, 0o755)

print(f"Batch scripts have been created in {SCRIPT_DIR}.")
