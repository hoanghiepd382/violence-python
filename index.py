import cv2
import os
import numpy as np
import glob
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def extract_frames(video_path, output_dir, num_frames=16, resize=(112, 112)):
    """
    Cắt num_frames frame từ 1 video và lưu vào thư mục
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Chọn khoảng cách giữa các frame
    step = max(total_frames // num_frames, 1)

    frames = []
    count = 0
    saved = 0

    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, resize)
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame)
            saved += 1
        count += 1

    cap.release()
    return np.array(frames)


# video_path = "test/V_118.mp4"
# output_dir = "datatest/frames"
# frames = extract_frames(video_path, output_dir, num_frames=16)
# print("Shape frames:", frames.shape)  # (16, 112, 112, 3)
# all_videos = glob.glob("data/RWF-2000/val/Fight/*.avi")
# for i, video_path in enumerate(all_videos):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     output_dir = f"dataset/val/frames/Fight/{video_name}"
#     extract_frames(video_path, output_dir, num_frames=16)
#     print(f"[{i + 1}/{len(all_videos)}] Done {video_name}")
#
# input_root = "dataset/test_frames/Fight"  # frames gốc
# output_root = "dataset/test_frames_resized/Fight"  # frames đã resize
# # Kích thước muốn resize
# target_size = (112, 112)
#
# for video_folder in os.listdir(input_root):
#     input_path = os.path.join(input_root, video_folder)
#     output_path = os.path.join(output_root, video_folder)
#
#     if not os.path.isdir(input_path):
#         continue
#
#     # Tạo folder output nếu chưa có
#     os.makedirs(output_path, exist_ok=True)
#
#     # Resize từng frame
#     for frame_file in os.listdir(input_path):
#         frame_in = os.path.join(input_path, frame_file)
#         frame_out = os.path.join(output_path, frame_file)
#
#         img = cv2.imread(frame_in)
#         if img is None:
#             continue
#         img_resized = cv2.resize(img, target_size)
#         cv2.imwrite(frame_out, img_resized)
#
#     print(f"✅ Done {video_folder}")

resnet = models.resnet18(pretrained=True)
# Bỏ đi layer cuối cùng (fc)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# 2. Transform để chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 3. Hàm trích xuất feature cho 1 frame
def extract_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # [1,3,112,112]
    with torch.no_grad():
        feature = feature_extractor(img)  # [1,512,1,1]
    return feature.view(-1)  # [512]


# 4. Trích xuất feature cho 1 folder frame (1 video)
def extract_features_from_video(folder_path):
    features = []
    for frame_name in sorted(os.listdir(folder_path)):
        frame_path = os.path.join(folder_path, frame_name)
        if not frame_path.lower().endswith((".jpg", ".png")):  # chỉ lấy ảnh
            continue
        feature = extract_feature(frame_path)
        features.append(feature)

    if len(features) == 0:
        print(f"[WARNING] Folder {folder_path} rỗng hoặc không có frame hợp lệ, bỏ qua.")
        return None  # hoặc trả về tensor rỗng

    return torch.stack(features)


#
#
# label_map = {"Fight": 1, "NonFight": 0}
# all_features = []
# all_labels = []
#
# test_features = []
# test_labels = []
#
# root_path = "dataset/train/frames"
# root_test = "dataset/val/frames"
#
# for class_name in os.listdir(root_path):
#     class_path = os.path.join(root_path, class_name)
#     if os.path.isdir(class_path):
#         label = label_map[class_name]
#
#         for video_folder in os.listdir(class_path):
#             folder_path = os.path.join(class_path, video_folder)
#             if os.path.isdir(folder_path):
#                 print(f"   --> Extract video: {video_folder}")
#                 features = extract_features_from_video(folder_path)
#                 if features is None:
#                     continue
#                 print(f"       Số frame: {features.shape[0]}, feature_dim: {features.shape[1]}")
#
#                 all_features.append(features)
#                 all_labels.append(label)
#
# x = torch.stack(all_features)  # [num_videos, num_frames, 512]
# y = torch.tensor(all_labels)
# torch.save((x, y), "features_dataset.pt")
#
# for class_name in os.listdir(root_test):
#     class_path = os.path.join(root_test, class_name)
#     if os.path.isdir(class_path):
#         label = label_map[class_name]
#
#         for video_folder in os.listdir(class_path):
#             folder_path = os.path.join(class_path, video_folder)
#             if os.path.isdir(folder_path):
#                 print(f"   --> Extract video: {video_folder}")
#                 features = extract_features_from_video(folder_path)
#                 if features is None:
#                     continue
#                 print(f"       Số frame: {features.shape[0]}, feature_dim: {features.shape[1]}")
#
#                 test_features.append(features)
#                 test_labels.append(label)
#
# x_test = torch.stack(test_features)
# y_test = torch.tensor(test_labels)
# torch.save((x_test, y_test), "features_test.pt")
