import torch.nn as nn


class VideoRNNClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, num_classes=2, rnn_type="LSTM"):
        super(VideoRNNClassifier, self).__init__()

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Chỉ hỗ trợ LSTM hoặc GRU")

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        out = self.fc(last_out)
        return out


class ViolenceRNN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, num_classes=2):
        super(ViolenceRNN, self).__init__()
        # LSTM nhận chuỗi feature
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected để phân loại
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [batch_size, num_frames, feature_dim]
        _, (hn, _) = self.lstm(x)  # hn shape: [num_layers, batch_size, hidden_dim]
        hn = hn[-1]  # lấy layer cuối: [batch_size, hidden_dim]
        out = self.fc(hn)  # [batch_size, num_classes]
        return self.softmax(out)
