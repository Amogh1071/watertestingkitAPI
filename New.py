import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def get_average_rgb(image, x, y, w, h):
    # Define the region of interest
    roi = image[y:y + h, x:x + w]

    # Get the average color in BGR
    avg_color_per_row = np.mean(roi, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return tuple(map(int, avg_color))  # Convert to integers

# Load and process reference tag images
reference_colors = {}
for tag_file in sorted(glob.glob('Tag*.png')):
    tag_img = cv2.imread(tag_file)
    if tag_img is not None:
        reference_colors[tag_file] = tag_img

# Load and process test image
test_image = cv2.imread('Images/sample 2(cropped).png')
if test_image is None:
    print("Error: Could not load test image")
    exit()

# Convert to HSV for processing
test_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

# Create mask for colored regions
lower = np.array([0, 30, 30])
upper = np.array([180, 255, 255])
mask = cv2.inRange(test_hsv, lower, upper)

# Apply morphological operations
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and sort contours
min_area = 50
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Process each region
output_image = test_image.copy()
color_data = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi_avg_color = get_average_rgb(test_image, x, y, w, h)

    # Store color information for plotting
    color_data.append((roi_avg_color, (x, y, w, h)))

    # Draw rectangle and label
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    tag_label = f"RGB {roi_avg_color}"

    # Center text
    text_size = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2

    cv2.putText(output_image, tag_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)

    print(f"Region at ({x}, {y}): RGB = {roi_avg_color}")

h,w= test_image.shape[:2]


# Save and display output image
cv2.imwrite('test_output.png', output_image)
output_image = cv2.resize(output_image, (int(w/2), int(h/2)))
cv2.imshow('test_output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot the colors
plt.figure(figsize=(10, 6))
for i, (color, rect) in enumerate(color_data):
    rgb = tuple(reversed(color))  # Convert BGR to RGB for plotting
    plt.bar(i, 1, color=np.array(rgb) / 255.0, edgecolor='black', label=f"RGB {rgb}")
    plt.text(i, 1.1, f"{rgb}", ha='center', fontsize=10)

plt.xticks([])
plt.yticks([])
plt.title("Detected Colors and RGB Values")
plt.show()
