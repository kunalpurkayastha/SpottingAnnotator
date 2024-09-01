import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define directories
images_dir = "images"
annotations_dir = "annotations"

# Function to plot points on the image
def plot_annotations(image_path, annotation_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    
    # Convert image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    # Plot each point
    for line in lines:
        points_text, detected_text = line.rsplit(',', 1)
        points = [tuple(map(float, pt.split(','))) for pt in points_text.split(', ')]

        # Plot each point
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=(255, 0, 0), thickness=-1)

        # Optionally, put the detected text near the first point
        cv2.putText(image, detected_text.strip(), (int(points[0][0]), int(points[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display the image with points
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Iterate through the annotations folder and plot each image
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith(".txt"):
        image_name = os.path.splitext(annotation_file)[0]
        image_path = os.path.join(images_dir, f"{image_name}.jpg")
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        # Plot the image with points
        plot_annotations(image_path, annotation_path)
