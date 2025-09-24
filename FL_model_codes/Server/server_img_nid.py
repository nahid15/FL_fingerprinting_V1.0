import socket
import pickle
import struct
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

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

# --- Improved Model Definition for CIFAR-10 ---

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

# --- Aggregation Function (Federated Averaging) ---
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
            # For non-floating point parameters (like num_batches_tracked),
            # simply take the value from the first update.
            global_state[k] = updates[0][k]
    return global_state


# --- Evaluation Function ---
def evaluate_model(model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# --- Main Server Loop ---
def main():
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 5000
    num_clients = 2   # Number of clients expected per round
    max_rounds = 3  # Increase number of rounds for better convergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the global model.
    global_model = ImprovedCNN()

    # Set up the server socket.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}")

    current_round = 1
    while current_round <= max_rounds:
        print(f"\n=== Starting round {current_round} ===")
        client_updates = []
        # Wait for the expected number of clients.
        for i in range(num_clients):
            print(f"Waiting for client {i+1}/{num_clients}...")
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            # Send the current global model and round number.
            msg = {"round": current_round, "global_model": global_model.state_dict()}
            send_msg(conn, msg)
            # Receive the client's updated model.
            update = recv_msg(conn)
            if update is None or update.get("round") != current_round:
                print("Received invalid update from client.")
            else:
                client_updates.append(update["client_model"])
                print(f"Received update from {addr}")
            conn.close()
        if client_updates:
            aggregated_state = aggregate_models(client_updates)
            global_model.load_state_dict(aggregated_state)
            print(f"Completed round {current_round}")
            # Evaluate the updated global model.
            loss, acc = evaluate_model(global_model, device)
            print(f"Evaluation after round {current_round}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        else:
            print("No valid updates received for this round.")
        current_round += 1

    print("Training complete. Shutting down server.")
    server_socket.close()

if __name__ == "__main__":
    main()
