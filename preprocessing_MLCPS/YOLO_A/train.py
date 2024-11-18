from ultralytics import YOLO
import os
import cv2

# TRAINING
# - For YOLO training, ensure the data and labels are arranged in the following specific folder structure:
#   yolo_A/
#     - data/
#         - images/
#             - train/
#             - val/
#         - labels/
#             - train/
#             - val/
# - Also, make sure to update the paths in the `config.yaml` file accordingly.
# - Add the path of the `config.yaml` here in the training script for proper configuration.

# Load the pre-trained YOLOv8 
model = YOLO("yolov8n.pt")  # Load pre-trained model

# Fine-tune the model on your custom dataset
results = model.train(data="............/yolo_A/config.yaml", epochs=40)

# PREDICTION ON TRAIN DATA TO REMOVE OUTLIERS
# - This step will predict on the training data and remove outliers.
# - Images without outliers will be saved in a separate folder for further use.
# Path to input images
input_image_path = "............./Train_Data/A"

# Path to save images that meet the confidence threshold
output_image_path = "............/Output_Conf_Images_A"  
os.makedirs(output_image_path, exist_ok=True)

# Iterate over each image in the input folder
for image_file in os.listdir(input_image_path):
    image_path = os.path.join(input_image_path, image_file)

    # Predict bounding boxes on the image without saving or drawing
    results = model(image_path, save=False)

    # Get the bounding box with the highest confidence
    if len(results[0].boxes) > 0:  # Ensure there is at least one detection
        
        filtered_boxes = [box for box in results[0].boxes if box.conf > 0.5]

        if filtered_boxes:  # If there are any boxes that satisfy the confidence condition
            # Save the original image in the output folder
            img = cv2.imread(image_path)
            output_file_path = os.path.join(output_image_path, image_file)
            cv2.imwrite(output_file_path, img)
            print(f"Saved image with confidence > 0.5: {output_file_path}")
        else:
            print(f"No boxes with confidence > 0.5 for image: {image_file}")
    else:
        print(f"No detection found for image: {image_file}")
