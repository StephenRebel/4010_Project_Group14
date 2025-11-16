import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from baseline_env_utils import EOS_ID, note_id_from_action
import os
import numpy as np

# Dataset representation of composition actions
class CompositionDataset(Dataset):
    def __init__(self, compositions):
        self.compositions = compositions

    def __len__(self):
        return len(self.compositions)
    
    def __getitem__(self, i):
        return torch.tensor(self.compositions[i], dtype=torch.long)
    
PAD = EOS_ID
save_path = "models/best_lstm.pth"

# Transforms batch into useable training form
def collate_fn(batch, pad_id):
    lengths = [b.size(0) for b in batch]
    max_len = max(lengths)
    padded_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        padded_batch[i, :lengths[i]] = b
    input_comp = padded_batch[:, :-1]
    target_comp = padded_batch[:, 1:]
    return input_comp, target_comp

class NextTokenLSTM(nn.Module):
    # Simple LSTM for sequence generation
    def __init__(self, vocab_size, embed_size=64, hidden_size=256, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, nlayers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden_x=None):
        e = self.embed(x)
        out, hidden_x = self.lstm(e, hidden_x)
        logits = self.fc(out)
        return logits, hidden_x
    
# Standard training loop
def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu', pad_id=EOS_ID):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    model.to(device)
    best_val = float('inf')
    os.makedirs(save_path, exist_ok=True)

    for ep in range(epochs):
        model.train()
        total = 0 
        count = 0
        for x in train_loader:
            x_in, x_target = x
            x_in = x_in.to(device)
            x_target = x_target.to(device)
            optim.zero_grad()
            logits, _ = model(x_in)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), x_target.view(B*T))
            loss.backward()
            optim.step()
            total += loss.item()
            count += 1
        val_loss = eval_loss(model, val_loader, device, pad_id)
        print(f"Epoch: {ep+1}, Train_loss={total/count:.3f} Val_loss={val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path))
    return model

def eval_loss(model, loader, device, pad_id):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total = 0
    n = 0
    with torch.no_grad():
        for x in loader:
            x_in, x_target = x
            x_in = x_in.to(device)
            x_target = x_target.to(device)
            logits, _ = model(x_in)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), x_target.view(B*T))
            total += loss.item()
            n += 1
    return total / max(1,n)

# Guarentee the same format ensured by the gymnasium environment. More difficult to capture with external models
def sample_from_lstm(model, eos_id, env=None, num_bars=None, temperature=1.0, device='cpu'):
    if env is None:
        raise ValueError("Must be provided a gymnasium environment for sampling")

    model.eval()
    model.to(device)

    out = []
    hidden_x = None # Initial hidden state
    current_bar = 0
    current_beats = 0.0
    beats_per_bar = env.beats_per_bar
    num_bars = env.bars
    vocab_size = model.fc.out_features

    # Start with a dummy initial input set to PAD or whatever used in training
    last = torch.tensor([[PAD]], dtype=torch.long, device=device)

    with torch.no_grad():
        # NOTE may need to consider hard limit but should end
        while current_bar < num_bars:

            logits, hidden_x = model(last, hidden_x)
            logits = logits[0, -1] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            remaining_beats = beats_per_bar - current_beats

            # Main filtering loop
            for i in range(vocab_size):
                if i == eos_id:
                    # Eos only for the end
                    if not (current_bar == num_bars - 1 and remaining_beats == 0):
                        probs[i] = 0
                else:
                    _, duration_id, _ = note_id_from_action(i, env)
                    duration = env.durations[duration_id]

                    # Filter valid actions, or moving to next bar
                    if remaining_beats > 0:
                        if duration > remaining_beats:
                            probs[i] = 0
                    else:
                        if current_bar == num_bars - 1:
                            probs[i] = 0

            # Handle no valid actions in predicted set
            if np.sum(probs) == 0:
                if current_bar == num_bars - 1 and remaining_beats == 0:
                    if eos_id is not None:
                        out.append(eos_id)
                    break
                else:
                    forced = []
                    for i in range(vocab_size):
                        pitch_id, duration_id, _ = note_id_from_action(i, env)
                        duration = env.durations[duration_id]
                        if pitch_id == env.rest_action and (remaining_beats == 0 or duration <= remaining_beats):
                            forced.append(i)
                    if forced:
                        prob_val = 1.0 / len(forced)
                        for i in forced:
                            probs[i] = prob_val
                    else:
                        probs = np.ones(vocab_size) / vocab_size

            # Normalize
            probs /= np.sum(probs) if np.sum(probs) > 0 else 1.0

            next_action = np.random.choice(vocab_size, p=probs)
            out.append(int(next_action))

            if eos_id is not None and next_action == eos_id:
                break

            last = torch.tensor([[next_action]], dtype=torch.long, device=device)

            # Update state
            _, duration_id, _ = note_id_from_action(next_action, env)
            duration = env.durations[duration_id]
            if current_beats + duration > beats_per_bar:
                current_bar += 1
                current_beats = 0.0
            current_beats += duration

    return out