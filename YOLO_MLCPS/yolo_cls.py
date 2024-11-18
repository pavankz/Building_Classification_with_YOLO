from ultralytics import YOLO

# For the YOLO model, ensure that the training and validation data folders are organized in the following specific path structure:
# YOLO_MLCPS/
#    - train/
#    - val/

# Load the YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")  # YOLOv8 Nano for classification

# Train the model on your dataset
model.train(
    data="...../YOLO_MLCPS",  # Path to the folder containing class subfolders
    epochs=100,  
    imgsz=224   # Image size 
)

# PREDICT ON TEST DATA
# - Make sure to update the `test_data_path` with the correct test data directory.
# - The results (.txt files for each image) will be saved in the path: YOLO_MLCPS/runs/classify/predict.
# - The trained model can be found at: /home/jai/ML_CPS_PROJECT_1/ml_cps_p1_d2/Project 1 Data 2/YOLO_MLCPS/runs/classify/train/weights/best.pt.  

#load the trained model
model = YOLO("/home/jai/ML_CPS_PROJECT_1/ml_cps_p1_d2/Train_Data_Complete/runs/classify/train/weights/best.pt")
# Path to the folder containing test images
test_data_path = "......../Test_Data"

# Make predictions on the test images
results = model.predict(source=test_data_path, conf=0.25, save=True, save_txt=True)

# Print the results
for result in results:
    print(f"Image: {result.path}")
    print(f"Predictions: {result.names}")  
    # print(f"Confidence Scores: {result.boxes.conf}")  
    print("---")


# To generate the submission.csv file for predictions:
# - Make sure to update the path for labels in the `labels_folder` (labels folder should be within the `predict` directory).
# - Also, specify the path where the CSV file should be downloaded.

import os
import csv

# Set the path for the labels folder and the output CSV file
labels_folder = '........../runs/classify/predict/labels'
output_csv = '/............/submission_yolo_detected.csv'


predictions = []


label_mapping = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'S': 5
}

# Iterate over each text file in the labels folder
for filename in sorted(os.listdir(labels_folder), key=lambda x: int(x.split('.')[0])):  
    if filename.endswith('.txt'):
        file_path = os.path.join(labels_folder, filename)

        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            
            max_score = 0.0
            predicted_class = None
            
            for line in lines:
                score, label = line.split()
                score = float(score)
                
                
                if score > max_score:
                    max_score = score
                    predicted_class = label

        
        image_id = os.path.splitext(filename)[0]
        
        
        predicted_numeric = label_mapping.get(predicted_class, None)

        if predicted_numeric is not None:
            predictions.append((int(image_id), predicted_numeric))


with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Predictions'])

    for image_id, predicted_numeric in predictions:
        writer.writerow([image_id, predicted_numeric])

print(f"Predictions saved to {output_csv}")
