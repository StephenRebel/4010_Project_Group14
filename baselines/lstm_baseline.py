import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json

from .baseline_env_utils import EOS_ID, note_id_from_action, model_vocab_mapping

# https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch
# https://colab.research.google.com/gist/SauravMaheshkar/168f0817f0cd29dd4048868fb0dd4401/lstms-in-pytorch.ipynb#scrollTo=Wuss5ZGQ9r0x
# https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

PAD = EOS_ID
save_path = "models/best_lstm.pth"
SEED = 42

# Dataset representation of composition actions
class CompositionDataset(Dataset):
    def __init__(self, compositions):
        self.compositions = compositions

    def __len__(self):
        return len(self.compositions)
    
    def __getitem__(self, i):
        return torch.tensor(self.compositions[i], dtype=torch.long)
    
# Transforms batch into useable training form
def collate_fn(batch):
    lengths = [b.size(0) for b in batch]
    max_len = max(lengths)
    padded_batch = torch.full((len(batch), max_len), PAD, dtype=torch.long)

    for i, b in enumerate(batch):
        padded_batch[i, :lengths[i]] = b

    # Model should learn to predict next token(s) given the previous one(s)
    input_comp = padded_batch[:, :-1]
    target_comp = padded_batch[:, 1:]
    return input_comp, target_comp

class LSTMMusicModel(nn.Module):
    # Simple LSTM for sequence generation
    def __init__(self, vocab_size, embed_size=64, hidden_size=256, dropout=0.2):
        super(LSTMMusicModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden_x=None):
        embeddings = self.embed(x)
        out, hidden_x = self.lstm(embeddings, hidden_x)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden_x
    
def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu', save_path="best_lstm.pth"):
    # Standard training loop for LSTM, save best model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    model.to(device)

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for x_input, x_target in train_loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_input)

            batch_size, time_steps, vocab_size = logits.shape
            loss = loss_fn(logits.reshape(batch_size*time_steps, vocab_size), x_target.reshape(batch_size*time_steps))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = eval_loss(model, val_loader, device)
        print(f"Epoch: {ep+1}/{epochs}, Train_loss={avg_train_loss:.3f} Val_loss={avg_val_loss:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

def eval_loss(model, loader, device):
    # Evaluate model loss after training epoch
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0

    with torch.no_grad():
        for x_input, x_target in loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)

            logits, _ = model(x_input)
            batch_size, time_steps, vocab_size = logits.shape
            loss = loss_fn(logits.reshape(batch_size*time_steps, vocab_size), x_target.reshape(batch_size*time_steps))

            total_loss += loss.item()

    return total_loss / len(loader)

# Guarentee the same format ensured by the gymnasium environment. More difficult to capture with external models
def sample_from_lstm(model, env=None, num_bars=None, temperature=1.0, device='cpu', random_seed=None):
    if env is None:
        raise ValueError("Must be provided a gymnasium environment for sampling")

    if random_seed is not None:
        np.random.seed(random_seed)

    model.eval()
    model.to(device)

    composition = []
    hidden_x = None # Initial hidden state, (h0, c0) in the examples but don't need to unpack

    current_bar = 0
    current_beat = 0.0
    beats_per_bar = env.beats_per_bar
    
    vocab_size = model.fc.out_features

    # Start with a dummy initial input set to PAD or whatever used in training
    last_action = torch.tensor([[PAD]], dtype=torch.long, device=device)

    with torch.no_grad():
        # NOTE may need to consider hard limit but should end
        while current_bar < num_bars:
            # Sample predictions from model
            logits, hidden_x = model(last_action, hidden_x)
            logits = logits[0, -1] / max(1e-8, temperature)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

            remaining_beats = beats_per_bar - current_beat

            # Filtering valid actions based on music composition structure
            valid_actions = np.zeros(vocab_size, dtype=bool)
            for action in range(vocab_size):
                if action == EOS_ID: continue

                _, duration_id, _ = note_id_from_action(action, env)
                duration = env.durations[duration_id]

                # Collection valid actions
                if duration <= remaining_beats:
                    valid_actions[action] = True

            # Ensure invalid actions set to 0 probability
            action_probs[~valid_actions] = 0.0

            # Fall back to random sample of valid actions if no valid prediction from model
            if np.sum(action_probs) == 0:
                possible_actions = valid_actions.astype(float)

                action_probs = possible_actions

            # Normalize to proper probability vector
            action_probs = action_probs / np.sum(action_probs)
            action = int(np.random.choice(vocab_size, p=action_probs))

            composition.append(action)
            last_action = torch.tensor([[action]], dtype=torch.long, device=device)

            _, duration_id, _ = note_id_from_action(action, env)
            action_duartion = env.durations[duration_id]
            current_beat += action_duartion

            if current_beat >= beats_per_bar:
                current_bar += 1
                current_beat = 0.0

    return composition

# Load and preprocess data appending EOS token for LSTM and creating train and validation split
def load_and_process_dataset(path):
    baseline_data = []

    with open(path, "r") as df:
        for line in df:
            json_line = json.loads(line)
            baseline_data.append(json_line["composition_actions"])

    processed_data, vocab_size = model_vocab_mapping(baseline_data)

    train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=SEED)
    return train_data, val_data, vocab_size

# Perform the model training so we can use it in other file
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 5e-3 # default for Adam Optimizer 1e-3
    EMBED_SIZE = 32
    HIDDEN_SIZE = 128
    DROPOUT = 0.4

    # Build dataset and loaders
    data_path = "baseline_dataset.jsonl"
    train_data, val_data, vocab_size = load_and_process_dataset(data_path)
    train_loader = DataLoader(CompositionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(CompositionDataset(val_data), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Creat and train model
    model = LSTMMusicModel(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
    trained_model = train_loop(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device, save_path=save_path)

