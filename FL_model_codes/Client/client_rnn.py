import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import argparse

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

# ----- Data Loading for Sunspots -----
def load_sunspots():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    df = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
    series = df["Sunspots"].values.astype(np.float32)
    return series

# ----- Function to Create Local Dataset -----
def create_local_dataset(full_series, train_size, seq_len, client_id, num_clients, mean, std):
    """
    Partition the full training series (first train_size points) among clients.
    Each client gets a contiguous segment.
    Normalize the local series using the provided mean and std.
    Returns a TensorDataset.
    """
    local_start = int(client_id * train_size / num_clients)
    local_end = int((client_id + 1) * train_size / num_clients)
    local_series = full_series[:train_size][local_start:local_end]
    norm_series = (local_series - mean) / std
    data_list = []
    target_list = []
    for i in range(len(norm_series) - seq_len):
        data_list.append(norm_series[i:i+seq_len])
        target_list.append(norm_series[i+seq_len])
    data_tensor = torch.tensor(data_list, dtype=torch.float32).unsqueeze(-1)
    target_tensor = torch.tensor(target_list, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(data_tensor, target_tensor)
    return dataset

# ----- Improved LSTM Forecasting Model (Same as Server) -----
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=12, output_size=1, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ----- Local Training Function -----
def local_train(model, device, train_loader, optimizer, criterion, local_epochs):
    model.train()
    for epoch in range(local_epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Local Epoch {epoch+1}/{local_epochs}, Loss: {avg_loss:.4f}")
    return model

# ----- Main Client Function -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0, help="Unique client identifier")
    parser.add_argument("--num_clients", type=int, default=3, help="Total number of clients")
    parser.add_argument("--num_rounds", type=int, default=5, help="Total number of federated rounds")
    args = parser.parse_args()
    client_id = args.client_id
    num_clients = args.num_clients
    #num_rounds = args.num_rounds
    num_rounds = 7
    
    SERVER_IP = "127.0.0.1"  # Change if the server is on a different machine.
    SERVER_PORT = 5000
    local_epochs = 7
    batch_size = 32
    learning_rate = 0.001
    seq_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the full sunspots series.
    full_series = load_sunspots()
    train_size = int(0.8 * len(full_series))
    
    # Loop for multiple federated rounds.
    for round_idx in range(num_rounds):
        print(f"\nClient {client_id}: Starting federated round {round_idx+1} of {num_rounds}")
        # Create a new socket connection for this round.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        
        # Receive the global model and associated parameters from the server.
        server_msg = recv_msg(sock)
        if server_msg is None or server_msg.get("round") is None:
            print("Failed to receive valid message from server.")
            sock.close()
            continue
        
        current_round = server_msg["round"]
        print(f"Client {client_id}: Received global model for round {current_round}")
        global_state = server_msg["global_model"]
        norm_params = server_msg["norm_params"]
        seq_len = server_msg["seq_len"]
        train_size = server_msg["train_size"]
        
        # Initialize local model and load global weights.
        model = LSTMForecast(input_size=1, hidden_size=64, num_layers=12, output_size=1, dropout=0.2).to(device) #64,8, 0.2
        model.load_state_dict(global_state)
        
        # Create the local dataset for this client.
        local_dataset = create_local_dataset(full_series, train_size, seq_len, client_id, num_clients,
                                               mean=norm_params["mean"], std=norm_params["std"])
        train_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up the local optimizer and loss.
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Perform local training.
        print(f"Client {client_id}: Training locally for round {current_round}...")
        model = local_train(model, device, train_loader, optimizer, criterion, local_epochs)
        
        # Send updated model parameters along with the round number back to the server.
        update_msg = {"round": current_round, "client_model": model.state_dict()}
        send_msg(sock, update_msg)
        sock.close()
        print(f"Client {client_id}: Completed round {current_round} and sent update.")
        
        # Optionally, wait a short time before the next round.
        time.sleep(1)

if __name__ == "__main__":
    main()
