from ultralytics import YOLO

# Load your YOLO helmet detection model
model = YOLO("yolo11m.pt")

# Print all class names (label map)
print("🚨 Class names in helmet_detection.pt:")
for idx, name in model.model.names.items():
    print(f"{idx}: {name}")
    print(model.model.names)

