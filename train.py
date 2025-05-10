import subprocess
import os

# Function to run a Python script
def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    #create a directory for the model if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('results'):
        os.makedirs('results')
    # Set environment variables for dataset path, batch size, epochs, and learning rate
# Run main.pratain.py first
    run_script('main_pretrain.py')
# Run main.finetune.py after main.pratain.py finishes
    run_script('main_finetune.py')
