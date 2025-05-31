import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torch.optim import AdamW
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Optional: for progress bars

# Import Evo2 specific components
try:
    from evo2 import Evo2
except ImportError:
    print("Evo2 library not found. Please ensure it is installed correctly.")
    # You might need to install it via: pip install . in the evo2-main directory
    # Or handle the import error appropriately
    exit()

# ---------------------------
# 1. 数据集定义
# ---------------------------
class Evo2RegressionDataset(Dataset):
    """
    Dataset for loading DNA sequences and expression values for Evo2 regression.
    Uses the tokenizer provided by the Evo2 model wrapper.
    """
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Args:
            csv_file (str): Path to the csv file with 'sequence' and 'expression' columns.
            tokenizer (object): The tokenizer instance from the Evo2 wrapper.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        try:
            self.data = pd.read_csv(csv_file)
            # Ensure columns exist
            if 'sequence' not in self.data.columns or 'expression' not in self.data.columns:
                raise ValueError("CSV file must contain 'sequence' and 'expression' columns.")
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            raise
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")

        seq = self.data.iloc[idx]['sequence']
        # Ensure sequence is a string
        if not isinstance(seq, str):
            seq = str(seq)

        target = float(self.data.iloc[idx]['expression'])

        # Tokenize the sequence using Evo2's tokenizer
        # Note: Evo2's CharLevelTokenizer might not need explicit padding/truncation args
        # in the same way as Hugging Face tokenizers. It typically handles tokenization
        # based on character-to-integer mapping. We need to manually pad/truncate.
        token_ids = self.tokenizer.tokenize(seq)

        # Manual Truncation/Padding
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(token_ids)
            # Assuming pad token id is 0, common for many tokenizers. Verify if needed.
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 1)
            token_ids.extend([pad_token_id] * padding_length)

        input_ids = torch.tensor(token_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            # No attention mask needed for Evo2
            'target': torch.tensor(target, dtype=torch.float)
        }

# ---------------------------
# 2. 模型定义：基于 Evo2 加回归头
# ---------------------------
class Evo2ForRegression(nn.Module):
    """
    Evo2 model with a regression head for predicting sequence activity.
    """
    def __init__(self, model_name="evo2_7b", dropout_rate=0.1, intermediate_size=512): # Added intermediate_size
        super().__init__()
        print(f"Loading Evo2 model: {model_name}...")
        # Use the Evo2 wrapper which handles loading and tokenizer
        # This might download the model if not found locally
        try:
            self.evo2_wrapper = Evo2(model_name=model_name, local_path='/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt')
        except Exception as e:
            print(f"Error loading Evo2 model '{model_name}': {e}")
            print("Please ensure the model name is correct and dependencies are installed.")
            raise

        # Freeze Evo2 base model parameters
        print("Freezing Evo2 base model parameters...")
        for param in self.evo2_wrapper.model.parameters():
            param.requires_grad = False

        # Get hidden size from the model config
        try:
            hidden_size = self.evo2_wrapper.model.config.hidden_size
        except AttributeError:
             # Fallback if config structure is different
             print("Warning: Could not automatically determine hidden_size from model config.")
             # You might need to hardcode this based on the model, e.g., 1920 for 1B, 4096 for 7B
             if '1b' in model_name:
                 hidden_size = 1920
             elif '7b' in model_name:
                 hidden_size = 4096
             elif '40b' in model_name:
                 hidden_size = 8192
             else:
                 raise ValueError("Cannot determine hidden size for unknown model.")
             print(f"Assuming hidden_size={hidden_size} based on model name.")

        # New fully connected layer
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # Regression head: Linear layer from intermediate_size to 1 (scalar prediction)
        self.regressor = nn.Linear(intermediate_size, 1)

        # Cast the new layers to bfloat16 to match the expected input dtype
        self.fc1.to(torch.bfloat16)
        self.regressor.to(torch.bfloat16)
        print(f"Evo2ForRegression initialized with hidden_size={hidden_size}, intermediate_size={intermediate_size}.")

    def forward(self, input_ids):
        """
        Forward pass through the Evo2 base model and the regression head.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs (batch_size, sequence_length).

        Returns:
            torch.Tensor: Regression predictions (batch_size,).
        """
        # Pass input through the Evo2 base model (handled by the wrapper)
        # We need the hidden states/embeddings, not just logits.
        # Use return_embeddings=True. We need a valid layer name.
        # Let's try the final normalization layer 'final_norm' or the last block output.
        layer_to_embed = "blocks.28.mlp.l3"
        #if hasattr(self.evo2_wrapper.model, 'norm_f'):
             # Try final norm layer if it exists (common in newer architectures)
             # The layer name needs to be known/registered for embedding extraction.
             # This might require inspecting the vortex/StripedHyena code or examples.
             # Let's tentatively use a name, assuming it's registered.
             # If 'final_norm' isn't registered, this will fail.
             # A safer bet might be the output of the last block.
             #layer_to_embed = f'blocks.{self.evo2_wrapper.model.config.num_layers - 1}'
             # layer_to_embed = 'final_norm' # Alternative guess

        if layer_to_embed:
            try:
                # Request embeddings from the specified layer
                _, embeddings = self.evo2_wrapper(input_ids, return_embeddings=True, layer_names=[layer_to_embed])
                # Shape: (batch, seq_len, hidden_size)
                last_layer_output = embeddings[layer_to_embed]
            except Exception as e:
                print(f"Warning: Could not get embeddings from layer '{layer_to_embed}'. Error: {e}")
                print("Falling back to using the raw output of the base model (before LM head).")
                # Fallback: Get raw output from the underlying StripedHyena model
                # This assumes the forward pass of the base model returns hidden states directly
                # or that the wrapper's default forward gives usable states.
                # The exact output format depends on the StripedHyena implementation.
                # Let's assume the wrapper's call returns (logits, maybe_hidden_states)
                # or just logits. If just logits, this approach won't work directly.
                # Let's try calling the base model directly.
                output = self.evo2_wrapper.model(input_ids) # Might return tuple or tensor
                if isinstance(output, tuple):
                    # Assuming the first element is the main sequence output
                    last_layer_output = output[0]
                else:
                    last_layer_output = output

                # Check if shape is (batch, seq_len, hidden_size)
                if len(last_layer_output.shape) != 3 or last_layer_output.shape[-1] != self.regressor.in_features:
                     print(f"Error: Output shape mismatch. Expected (*, *, {self.regressor.in_features}), got {last_layer_output.shape}")
                     print("Cannot proceed with regression head. Check model output format.")
                     # You might need to adapt how embeddings are extracted based on Evo2/Vortex specifics.
                     # Returning zeros as a placeholder to avoid crashing, but this needs fixing.
                     return torch.zeros(input_ids.shape[0], device=input_ids.device)

        else:
             print("Error: Could not determine a layer to extract embeddings from.")
             return torch.zeros(input_ids.shape[0], device=input_ids.device)


        # Pool the hidden states across the sequence length dimension
        # Use mean pooling: average embeddings across the sequence length
        # Shape: (batch, hidden_size)
        pooled_output = last_layer_output.mean(dim=1)

        # Pass through the new fully connected layer and activation
        x = self.fc1(pooled_output)
        x = self.activation(x)

        # Apply dropout and the regression head
        x = self.dropout(x)
        prediction = self.regressor(x)

        # Squeeze the output to get shape (batch_size,)
        return prediction.squeeze(1)

# ---------------------------
# 3. 训练与验证函数
# ---------------------------
def train_epoch(model, data_loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    # Wrap data_loader with tqdm for a progress bar (optional)
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device, dtype=torch.bfloat16)

        # Forward pass
        outputs = model(input_ids)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        # Update progress bar description (optional)
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

def evaluate(model, data_loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Disable gradient calculations
        # Wrap data_loader with tqdm for a progress bar (optional)
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device, dtype=torch.bfloat16)

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * input_ids.size(0)
            # Update progress bar description (optional)
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

# ---------------------------
# 4. 主程序入口
# ---------------------------
def main():
    # --- 配置参数 ---
    # Model choice: 'evo2_1b_base', 'evo2_7b', 'evo2_7b_base', 'evo2_40b', 'evo2_40b_base'
    # Larger models require more GPU memory. Start with 'evo2_1b_base'.
    model_name = "evo2_7b"
    # Path to your data file
    # Make sure this path is correct and accessible
    # Example: "data/dna_activity.csv" or "/content/drive/MyDrive/sitp/data.csv"
    csv_file = "data.csv" # <--- *** UPDATE THIS PATH ***
    output_dir = "evo2_regression_output" # Directory to save the model
    output_model_file = os.path.join(output_dir, f"{model_name}_regression.pt")

    max_length = 512   # Max sequence length for tokenizer
    batch_size = 8     # Adjust based on GPU memory (Evo2 can be memory intensive)
    epochs = 100        # Number of training epochs
    learning_rate = 1e-5 # Learning rate for the regression head
    train_split = 0.8  # Proportion of data for training
    dropout_rate = 0.1 # Dropout rate for the regression head
    intermediate_hidden_size = 512 # Size of the new intermediate hidden layer

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == torch.device("cpu"):
        print("Warning: Running on CPU. Training may be very slow.")

    # --- 加载模型和分词器 ---
    # The Evo2ForRegression class handles loading the base model and tokenizer
    try:
        model = Evo2ForRegression(
            model_name=model_name,
            dropout_rate=dropout_rate,
            intermediate_size=intermediate_hidden_size # Pass intermediate_size
        )
        tokenizer = model.evo2_wrapper.tokenizer # Get tokenizer from the wrapper
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return # Exit if model loading fails

    model.to(device)

    # --- 构建数据集 ---
    try:
        dataset = Evo2RegressionDataset(csv_file, tokenizer, max_length)
    except (FileNotFoundError, ValueError) as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure the csv_file path is correct and the file is valid.")
        return # Exit if dataset loading fails

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    try:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    except ValueError as e:
        print(f"Error splitting dataset (train_size={train_size}, val_size={val_size}, total={len(dataset)}): {e}")
        print("Ensure your dataset has enough samples for the split.")
        return


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity, adjust if needed
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- 设置优化器和损失函数 ---
    # Only optimize the parameters of the regression head (fc1 and regressor)
    optimizer = AdamW(
        list(model.fc1.parameters()) + list(model.regressor.parameters()),
        lr=learning_rate
    )
    criterion = nn.MSELoss() # Mean Squared Error loss for regression

    # --- 开始训练 ---
    best_val_loss = float('inf') # Initialize with infinity

    print("Starting training...")
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on the validation set
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only the state_dict of the regression head, or the whole model
            # Saving the whole model state_dict is easier for reloading
            torch.save(model.state_dict(), output_model_file)
            print(f"Validation loss improved. Model saved to {output_model_file}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {output_model_file}")

if __name__ == "__main__":
    # IMPORTANT: Update the 'csv_file' variable in the main() function
    # to point to your actual data file before running.
    main()