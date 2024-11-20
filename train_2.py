import argparse
import os
import torch
import numpy as np
import csv
from model import GPTConfig, GPT

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train GPT model with varying embedding sizes and datasets.')
parser.add_argument('--n_embd', type=int, required=True, help='Size of the embedding')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--compile', type=bool, default=False, help='Whether to compile the model')
args = parser.parse_args()

# Print the configuration for debugging
print(f"n_embd: {args.n_embd}, dataset: {args.dataset}")

# Update the configuration
config = GPTConfig(
    n_layer=6,
    n_head=2,  # Ensure this matches the value in run_multiple_models.py
    n_embd=args.n_embd,
    dropout=0.0,
    block_size=512
)

# Print the configuration for debugging
print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, dropout={config.dropout}, block_size={config.block_size}")

# Initialize the model
model = GPT(config)

# Create results CSV file if it doesn't exist
results_file = 'results.csv'
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Embedding Size', 'Training Loss', 'Validation Loss'])

# Function to get batches of data
def get_batch(split, batch_size, block_size, data_dir):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

# Function to train and evaluate the model
def train_and_evaluate(n_embd, dataset):
    # Update the configuration
    config = GPTConfig(
        n_layer=6,
        n_head=2,
        n_embd=n_embd,
        dropout=0.0,
        block_size=1024
    )
    model = GPT(config)

    # Set dataset-specific parameters
    data_dir = os.path.join('data', dataset)
    batch_size = 32
    block_size = 1024

    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type='cuda')

    # Training loop
    num_epochs = 10  # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()
        for iteration in range(100):  # Adjust the number of iterations per epoch
            x, y = get_batch('train', batch_size, block_size, data_dir)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            # Print iteration count and loss
            print(f"Iteration: {iteration}, Loss: {loss.item()}")

        # Evaluate the model
        train_loss = estimate_loss(model, 'train', batch_size, block_size, data_dir)
        val_loss = estimate_loss(model, 'val', batch_size, block_size, data_dir)

        print(f"n_embd: {n_embd}, dataset: {dataset}, epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")

        # Save results to CSV
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset, n_embd, train_loss, val_loss])

    # Save the model
    model_dir = os.path.join('models', dataset, f'embedding_{n_embd}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

# Function to estimate loss
def estimate_loss(model, split, batch_size, block_size, data_dir):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):  # Adjust the number of evaluation iterations
            x, y = get_batch(split, batch_size, block_size, data_dir)
            logits, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# Main function
if __name__ == "__main__":
    train_and_evaluate(args.n_embd, args.dataset)