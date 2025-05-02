from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog
import os

# Step 1: Open file picker to choose image
Tk().withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not image_path:
    print("No image selected.")
    exit()

# Step 2: Load the model
model = YOLO("Brick_Model_best20250123_192838t.pt")

# Step 3: Run prediction
results = model(image_path)

# Step 4: Draw bounding boxes on the image
image = cv2.imread(image_path)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()     # Bounding boxes
    classes = result.boxes.cls.cpu().numpy()    # Class IDs
    confs = result.boxes.conf.cpu().numpy()     # Confidences

    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}" if model.names else f"{int(cls)} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Step 5: Save the result
# Generate output filename by adding '_detected' before the extension
filename, extension = os.path.splitext(image_path)
output_path = f"{filename}_detected{extension}"

# Save the image
cv2.imwrite(output_path, image)
print(f"Image saved as: {output_path}")