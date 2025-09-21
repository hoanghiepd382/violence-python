import torch
import torch.nn.functional as F
from model import VideoRNNClassifier
from index import extract_features_from_video  # hàm bạn viết ở index.py

# 1. Load model
model = VideoRNNClassifier(input_dim=512, hidden_dim=256, num_layers=1, num_classes=2)
model.load_state_dict(torch.load("violence_rnn.pth"))
model.eval()

# 2. Trích feature từ video mới
video_folder = "datatest/frames"  # hoặc folder chứa frame
features = extract_features_from_video(video_folder)  # [num_frames, 512]
print(features)
features = features.unsqueeze(0)  # [1, num_frames, 512] để batch_size=1
print(features)
# 3. Dự đoán
with torch.no_grad():
    logits = model(features)               # [1,2]
    probs = F.softmax(logits, dim=1)       # xác suất
    pred = probs.argmax(dim=1).item()      # nhãn dự đoán

labels = {0: "NonFight", 1: "Fight"}
print("Predicted label:", labels[pred])
print("Probabilities:", probs.tolist())
