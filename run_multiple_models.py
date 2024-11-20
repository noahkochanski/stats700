import subprocess

# Define the range of n_embd values and datasets
n_embd_values = [24, 48]  # Ensure these values are divisible by n_head (2)
datasets = ['shakespeare_char']

# Define the number of heads (update this value based on your model configuration)
n_head = 2  # Ensure this matches the value in your model configuration

# Define the base command to run the train.py script
base_command = ["python", "train_2.py"]

# Iterate over each combination of n_embd and dataset
for n_embd in n_embd_values:
    # Ensure n_embd is divisible by n_head
    if n_embd % n_head != 0:
        print(f"Skipping n_embd={n_embd} as it is not divisible by n_head={n_head}")
        continue
    
    for dataset in datasets:
        # Construct the command with the specific parameters
        command = base_command + [
            f"--n_embd={n_embd}",
            f"--dataset={dataset}",
            "--batch_size=32",
            "--compile=False"
        ]
        
        print(f"Running command: {' '.join(command)}")
        # Execute the command
        subprocess.run(command)