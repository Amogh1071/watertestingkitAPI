from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")

# Load the image
image_path = 'Images/sample 2.jpg'
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the original dimensions of the image
h, w, _ = image.shape

# Define the target size (e.g., 640x640)
target_size = 640

# Calculate the scale factor to maintain the aspect ratio
scale = target_size / max(h, w)

# Compute new dimensions based on the scale
new_w = int(w * scale)
new_h = int(h * scale)

# Resize the image while maintaining the aspect ratio
resized_image = cv2.resize(image, (new_w, new_h))

# Create a new image of the target size and place the resized image in the center
padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255  # white padding
pad_x = (target_size - new_w) // 2
pad_y = (target_size - new_h) // 2

# Place the resized image in the center of the padded image
padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

# Make predictions using the padded image
result = model(padded_image)

# List to store valid RGB colors and their centers for plotting
valid_colors = []

# Draw bounding boxes and calculate average RGB values
for bbox in result[0].boxes:
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
    center_x = int((x1 + x2) /2)
    center_y = int ((y1 + y2) / 2)

    # Get the corners and center points
    points = [
        # (x1, y1),  # Top-left
        # (x2, y1),  # Top-right
        # (x1, y2),  # Bottom-left
        # (x2, y2),  # Bottom-right
        (center_x, center_y)  # Center
    ]

    # Calculate average RGB values
    avg_rgb = np.mean([padded_image[pt[1], pt[0]] for pt in points], axis=0).astype(int)

    # Check if the color is not white
    if not (avg_rgb[0] > 180 and avg_rgb[1] > 180 and avg_rgb[2] > 180):  # R > 180, G > 180, B > 180
        valid_colors.append((center_y, avg_rgb))  # Append the y-coordinate and RGB value

        # Draw bounding box
        cv2.rectangle(padded_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display RGB values as text near the bounding box
        text = f"RGB: {avg_rgb.tolist()}"
        cv2.putText(padded_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Sort valid colors by y-coordinate (ascending)
valid_colors.sort(key=lambda x: x[0])

# Extract only the RGB values for plotting
sorted_colors = [data[1] for data in valid_colors]

# Create a color visualization
plt.figure(figsize=(5, 10))
for i, rgb in enumerate(sorted_colors):
    plt.barh(i, 1, color=np.array(rgb) / 255, edgecolor='black')  # Normalize RGB to [0, 1]

plt.title("Detected Colors (From Top to Bottom)")
plt.yticks(range(len(sorted_colors)), [f"Color {i+1}" for i in range(len(sorted_colors))])
plt.xticks([])
plt.gca().invert_yaxis()  # Top-to-bottom ordering
plt.show()

padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes and RGB Values', padded_image)

# Wait for a key press and close OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
