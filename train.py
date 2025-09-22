from model import VideoRNNClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load dữ liệu đã extract feature
x, y = torch.load("features_dataset.pt")

# Đưa dữ liệu vào Dataset và DataLoader với batch_size = 4
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Khởi tạo model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoRNNClassifier(input_dim=512, hidden_dim=256, num_classes=2, rnn_type="LSTM").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

# Lưu model
torch.save(model.state_dict(), "violence_rnn.pth")
