import torch

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
output_path = "yolovl.pt"

print(f"Downloading YOLOv8 Large weights from {url} ...")
torch.hub.download_url_to_file(url, output_path)
print(f"✅ Download complete! Saved as {output_path}")
