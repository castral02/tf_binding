# this code is an example of how to run all the SLURMs in one folder so you do not have to individually write out sbatch ____.sh
import os
import subprocess

# Define the directory where your SLURM scripts are stored
scripts_dir = "generated_pulldown_scripts"

# Get a list of all .sh files in the directory
scripts = [f for f in os.listdir(scripts_dir) if f.endswith(".sh")]

# Sort scripts to ensure they run in a predictable order (optional)
scripts.sort()

# Submit each script using sbatch
for script in scripts:
    script_path = os.path.join(scripts_dir, script)
    try:
        # Run the sbatch command
        subprocess.run(["sbatch", script_path], check=True)
        print(f"Submitted {script}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit {script}: {e}")

