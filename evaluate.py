import torch
import torch.nn.functional as F
from model import VideoRNNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load model đã train
model = VideoRNNClassifier(input_dim=512, hidden_dim=256, num_layers=1, num_classes=2)
model.load_state_dict(torch.load("violence_rnn.pth"))  # file bạn lưu model sau train
model.eval()  # chuyển sang chế độ evaluate

# Load dữ liệu đánh giá (test set)
x_test, y_test = torch.load("features_test.pt")
print("X_val:", x_test.shape, "Y_val:", y_test.shape)

with torch.no_grad():
    logits = model(x_test)             # [num_videos, 2]
    probs = F.softmax(logits, dim=1)   # xác suất cho từng class
    preds = probs.argmax(dim=1)        # nhãn dự đoán 0/1

# Tính metrics
y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# In thử 5 video đầu tiên
for i in range(min(5, len(y_true))):
    print(f"Video {i}: Pred={y_pred[i]}, Prob={probs[i].tolist()}, True={y_true[i]}")
