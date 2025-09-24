import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Helper Functions for Socket Communication -----
def send_msg(sock, msg):
    data = pickle.dumps(msg)
    msg_len = struct.pack('>I', len(data))
    sock.sendall(msg_len + data)

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = recvall(sock, msglen)
    return pickle.loads(data)

# ----- Data Loading and Preprocessing -----
def load_sunspots():
    """
    Downloads and loads the monthly sunspots dataset.
    Returns the sunspot numbers as a 1D numpy array of type float32.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    df = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
    series = df["Sunspots"].values.astype(np.float32)
    return series

class SunspotsDataset:
    def __init__(self, series, seq_len, mean, std):
        """
        Constructs input/output pairs from the normalized sunspots series.
        The series should already be normalized using the provided mean and std.
        """
        self.seq_len = seq_len
        self.data = []
        self.targets = []
        for i in range(len(series) - seq_len):
            self.data.append(series[i:i+seq_len])
            self.targets.append(series[i+seq_len])
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(-1)
    
    def __len__(self):
        return len(self.data)
    
    def get_loader(self, batch_size, shuffle=False):
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(self.data, self.targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ----- Improved LSTM Forecasting Model -----
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=12, output_size=1, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ----- Evaluation Function (Centralized on Test Set) -----
def evaluate_global(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

# ----- Aggregation Function (Federated Averaging) -----
def aggregate_models(updates):
    global_state = None
    for state in updates:
        if global_state is None:
            global_state = {k: torch.zeros_like(v) for k, v in state.items()}
        for k in global_state:
            global_state[k] += state[k]
    for k in global_state:
        if torch.is_floating_point(global_state[k]):
            global_state[k] /= len(updates)
        else:
            global_state[k] = updates[0][k]
    return global_state

# ----- Main Federated Server Function -----
def main():
    HOST = "0.0.0.0"
    PORT = 5000
    NUM_CLIENTS = 3
    FED_ROUNDS = 7  # number of federated rounds
    LOCAL_EPOCHS = 5
    seq_len = 32
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load full sunspots series and split into train/test.
    series = load_sunspots()
    print("Full series length:", len(series))
    train_size = int(0.8 * len(series))
    train_series = series[:train_size]
    test_series = series[train_size - seq_len:]
    
    # Compute normalization parameters on training series.
    train_mean = np.mean(train_series)
    train_std = np.std(train_series)
    
    # Normalize full training series.
    norm_train_series = (train_series - train_mean) / train_std
    norm_test_series = (test_series - train_mean) / train_std
    
    # Create test dataset and loader.
    test_dataset = SunspotsDataset(norm_test_series, seq_len, train_mean, train_std)
    test_loader = test_dataset.get_loader(batch_size, shuffle=False)
    
    # Initialize global model and move to device.
    global_model = LSTMForecast(input_size=1, hidden_size=64, num_layers=12, output_size=1, dropout=0.2).to(device)
    
    # Set up server socket.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}")
    
    # Federated training loop.
    for round_num in range(1, FED_ROUNDS + 1):
        print(f"\n=== Federated Round {round_num} ===")
        client_updates = []
        for client in range(NUM_CLIENTS):
            print(f"Waiting for client {client+1}/{NUM_CLIENTS}...")
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            # Send global model (converted to CPU) and additional parameters.
            msg = {
                "round": round_num,
                "global_model": {k: v.cpu() for k, v in global_model.state_dict().items()},
                "norm_params": {"mean": float(train_mean), "std": float(train_std)},
                "seq_len": seq_len,
                "train_size": train_size
            }
            send_msg(conn, msg)
            # Receive client's updated model.
            update = recv_msg(conn)
            if update is None or update.get("round") != round_num:
                print("Invalid update from client.")
            else:
                client_updates.append(update["client_model"])
                print(f"Received update from {addr}")
            conn.close()
        if client_updates:
            aggregated_state = aggregate_models(client_updates)
            global_model.load_state_dict(aggregated_state)
            print(f"Aggregated global model at round {round_num}")
            # Evaluate the updated global model.
            test_loss = evaluate_global(global_model, device, test_loader, nn.MSELoss())
            print(f"Round {round_num} Test MSE Loss: {test_loss:.4f}")
        else:
            print("No client updates received in this round.")
    
    print("Federated training complete.")
    
    # Optionally: Plot predictions vs. actual for test set.
    global_model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = global_model(data)
            predictions.extend(output.cpu().numpy().flatten())
            actual.extend(target.cpu().numpy().flatten())
    
    # Denormalize predictions.
    predictions = np.array(predictions) * train_std + train_mean
    actual = np.array(actual) * train_std + train_mean
    
    plt.figure(figsize=(10,5))
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Sunspot Number")
    plt.title("Federated Sunspots Forecasting")
    plt.legend()
    plt.show()
    
    server_socket.close()

if __name__ == "__main__":
    main()
