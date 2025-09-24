import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import time
import argparse

# --- Socket Communication Helpers ---
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

# --- Improved Model Definition (must match the server's) ---
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Input 3x32x32 -> Output 64 channels, spatial dims reduced to 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64 channels -> 128 channels, spatial dims reduced to 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 128 channels -> 256 channels, spatial dims reduced to 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        

# --- Function to Partition CIFAR-10 in a Non-IID Fashion ---
def get_non_iid_indices(dataset, client_id, num_clients):
    """
    For 2 clients:
      - Client 0 gets samples with labels 0–4.
      - Client 1 gets samples with labels 5–9.
    For more clients, adjust accordingly.
    """
    if num_clients == 2:
        if client_id == 0:
            allowed = set(range(5))  # classes 0,1,2,3,4
        elif client_id == 1:
            allowed = set(range(5, 10))  # classes 5,6,7,8,9
        indices = [i for i, label in enumerate(dataset.targets) if label in allowed]
        return indices
    else:
        # For more clients, use a round-robin partition over classes.
        classes = list(range(10))
        allowed = set(classes[client_id::num_clients])
        indices = [i for i, label in enumerate(dataset.targets) if label in allowed]
        return indices

# --- Local Training Function ---
def train_local(model, dataloader, epochs, lr, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    return model

# --- Main Client Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0, help="Unique client identifier")
    parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients")
    args = parser.parse_args()
    client_id = args.client_id
    num_clients = args.num_clients

    SERVER_IP = '127.0.0.1'  # Change to the server's IP if running on a different machine
    SERVER_PORT = 5000
    rounds = 3      # Increase rounds for better convergence
    local_epochs = 3      # Increase local epochs per round
    lr = 0.001             # Adjust learning rate for Adam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # Load the full CIFAR-10 training dataset.
    full_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    # Create a non-IID partition.
    indices = get_non_iid_indices(full_dataset, client_id, num_clients)
    print(f"Client {client_id} using {len(indices)} samples out of {len(full_dataset)}")
    client_dataset = Subset(full_dataset, indices)
    train_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

    # Initialize the local model.
    local_model = ImprovedCNN().to(device)

    for current_round in range(1, rounds + 1):
        print(f"\n--- Round {current_round}: Client {client_id} connecting to server... ---")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        # Receive global model and round info.
        msg = recv_msg(sock)
        if msg is None or msg.get("round") != current_round:
            print("Received invalid message from server.")
            sock.close()
            continue
        print("Received global model from server.")
        local_model.load_state_dict(msg["global_model"])

        # Train locally on non-IID CIFAR-10 data.
        print("Training locally on non-IID data...")
        local_model = train_local(local_model, train_loader, local_epochs, lr, device)
        # Simulate training delay.
        time.sleep(2)
        # Send updated model parameters back to the server.
        update_msg = {"round": current_round, "client_model": local_model.state_dict()}
        serialized = pickle.dumps(update_msg)
        print(f"update_msg size: {len(serialized)} bytes")
        send_msg(sock, update_msg)
        sock.close()
        print(f"Round {current_round} completed for client {client_id}.")

    print("Local training complete.")

if __name__ == "__main__":
    main()
