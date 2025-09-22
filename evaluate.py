import torch
import torch.nn.functional as F
from model import VideoRNNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load model đã train
model = VideoRNNClassifier(input_dim=512, hidden_dim=256, num_layers=1, num_classes=2)
model.load_state_dict(torch.load("violence_rnn.pth"))
model.eval()

# Load dữ liệu đánh giá (test set)
x_test, y_test = torch.load("features_test.pt")
print("X_test:", x_test.shape, "Y_test:", y_test.shape)

# Kiểm tra shape dữ liệu
assert len(x_test.shape) == 3 and x_test.shape[1] == 16 and x_test.shape[2] == 512, "Invalid X_test shape"
assert len(y_test.shape) == 1, "Invalid Y_test shape"

with torch.no_grad():
    logits = model(x_test)             # [num_videos, 2]
    probs = F.softmax(logits, dim=1)   # Xác suất cho từng class
    preds = probs.argmax(dim=1)        # Nhãn dự đoán 0/1

# Chuyển về numpy
y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()

# === Kết quả tổng thể ===
acc = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average="macro")
rec_macro = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")

print("\n===== Kết quả tổng thể =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")
print(f"Recall (macro)   : {rec_macro:.4f}")
print(f"F1-score (macro) : {f1_macro:.4f}")

# === Kết quả riêng lớp Violence (nhãn 1) ===
pre_violence = precision_score(y_true, y_pred, pos_label=1)
rec_violence = recall_score(y_true, y_pred, pos_label=1)
f1_violence = f1_score(y_true, y_pred, pos_label=1)

print("\n===== Kết quả riêng lớp Violence =====")
print(f"Precision (Violence): {pre_violence:.4f}")
print(f"Recall (Violence)   : {rec_violence:.4f}")
print(f"F1-score (Violence) : {f1_violence:.4f}")

# === Report chi tiết 2 lớp ===
print("\n===== Báo cáo chi tiết từng lớp =====")
print(classification_report(y_true, y_pred, target_names=["non_violence", "violence"]))