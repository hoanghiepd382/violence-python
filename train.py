from model import VideoRNNClassifier
import torch
import torch.nn as nn

# Load dữ liệu đã extract feature
x, y = torch.load("features_dataset.pt")

# Khởi tạo model
model = VideoRNNClassifier(input_dim=512, hidden_dim=256, num_classes=2, rnn_type="LSTM")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "violence_rnn.pth")
