import os

# Set the parent directory containing the folders
PARENT_DIR = "/data/CBLCCBR/crebbp_ep300/crebbp"  # Change this to your actual directory path
SCRIPT_DIR = os.path.join(PARENT_DIR, "scripts/generated_colab_scripts")

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
#SBATCH --job-name={folder_name}_colab       # Give your job a name
#SBATCH --partition=norm                # Specify the correct GPU partition
#SBATCH --mem=200G                      # Request 256GB of memory (adjust as necessary)
#SBATCH --cpus-per-task=16              # Request 32 CPU cores (adjust as necessary)
#SBATCH --gres=lscratch:50             # Request 100GB of local scratch space               
#SBATCH --time=3-24:00:00                   # Set a time limit for the job (6 hours)
#SBATCH --output=output/{folder_name}_colab.log    # Unique output log for each job
#SBATCH --error=error/{folder_name}_colab.log      # Unique error log for each job
#SBATCH --mail-type=END,FAIL   # Send email on job completion (END) or failure (FAIL)
#SBATCH --mail-user=  # Replace with your email address

set -e
set -x  # Enable debugging

# Load necessary modules
module load colabfold alphapulldown

# Specify your input and output directories
INPUT_DIR="{folder_path}"   # Set input directory dynamically
OUTPUT_DIR="{folder_path}"      # Set output directory dynamically

# Create individual features
create_individual_features.py --fasta_paths=$INPUT_DIR/bait.fasta,$INPUT_DIR/candidate.fasta --output_dir=$OUTPUT_DIR/pulldown_cf_msas --use_precomputed_msas=True --max_template_date=2023-01-01 --use_mmseqs2=True --skip_existing=True
""")

        # Make the script executable
        os.chmod(script_path, 0o755)

print(f"Batch scripts have been created in {SCRIPT_DIR}.")

