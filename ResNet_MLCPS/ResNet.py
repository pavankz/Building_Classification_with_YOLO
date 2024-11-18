import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model


import os

import matplotlib.pyplot as plt
from IPython.display import Image
from PIL import Image
import seaborn as sns

#preprocessing
'''

   please specify the data for traing in  train_dir and val_dir I TOOK DATA FOR THIS PART FROM YOLO_MLCPS folder make sure to chnage the path
   this code will convert images into numbers to train the model

'''

import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

# Paths to the train and val directories
train_dir = '..........YOLO_MLCPS/train'
val_dir = '............YOLO_MLCPS/val'

# Label mapping
label_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'S': 5}

image_size = (244, 244)

X_train, y_train = [], []
X_test, y_test = [], []

# Function to process images in a directory (train or val)
def process_images(data_dir):
    X, y = [], []  
    
    for folder, label in label_map.items():
        folder_path = os.path.join(data_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue

        sorted_filenames = sorted(os.listdir(folder_path))
        
        for filename in sorted_filenames:
            if filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    image = Image.open(img_path).convert('L')  
                    image = image.resize(image_size)  

                    
                    image_array = np.array(image)

                    
                    if image_array.shape == (244, 244):
                        X.append(image_array)  # Store the image as 2D array (244, 244)
                        y.append(label)
                    else:
                        print(f"Skipping {filename}: Incorrect image dimensions {image_array.shape}")
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    return np.array(X), np.array(y)

X_train, y_train = process_images(train_dir)
X_test, y_test = process_images(val_dir)

# Reshape the data
X_train = X_train.reshape(-1, 244, 244, 1)  # Adding a channel dimension (1 for grayscale)
X_train = np.repeat(X_train, 3, axis=-1)  # Repeat channel to get 3-channel images

X_test = X_test.reshape(-1, 244, 244, 1)  
X_test = np.repeat(X_test, 3, axis=-1)  

# One-hot encoding of labels
y_train = to_categorical(y_train - 1)  # Adjust index to start from 0
y_test = to_categorical(y_test - 1)    

# Check the shapes
print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Validation data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

#training the model

#class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights_dict = dict(enumerate(class_weights))

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

#pre-trained model as a base (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the pre-trained model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax') 
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=30,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# from tensorflow.keras.models import save_model
# model.save('/home/jai/ML_CPS_PROJECT_1/ml_cps_p1_d2/Project 1 Data 2/models/ResNet244_detected(1).h5')  

# print("Model saved successfully!")


# Predict labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = np.argmax(y_test, axis=1)  

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  

# Plot Normalized Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training & validation accuracy over epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#load the finetuned model on our dataset
# # Load the model from the file
# loaded_model = load_model('/home/jai/ML_CPS_PROJECT_1/ml_cps_p1_d2/ResNet_MLCPS/ResNet244_detected(1).h5')

# print("Model loaded successfully!")

#to convert test images into numbers
#'''
#  make sure to spacify Test_Data path in data_dir
#'''

data_dir = '............/Test_Data'

image_data = []
ids = []  


image_size = (244, 244)  


sorted_filenames = sorted(os.listdir(data_dir))

for filename in sorted_filenames:
    if filename.endswith('.jpg'):
        
        image_id = int(filename.split('.')[0])  
        img_path = os.path.join(data_dir, filename)
        image = Image.open(img_path)
        image = image.resize(image_size)
        image_array = np.array(image)
        
        image_array = image_array.flatten()
        
        
        image_data.append(image_array)
        ids.append(image_id)


image_data = np.array(image_data)
image_data = image_data.astype(float)


df = pd.DataFrame(image_data)


df['ID'] = ids 
test_data = df.sort_values(by='ID').reset_index(drop=True)
test_data

# output_csv_path = '/home/ravindra/MLCPS_PROJECT_1/Project 1 Data 2/testcombined_image_data.csv'
# df.to_csv(output_csv_path, index=False)

# print("Data processing complete. Data saved to 'testcombined_image_data.csv'.")

#prediction on test data  please specify the output path for getting output as csv

test_X= test_data.iloc[:, :-1].values 
test_X = test_X.reshape(-1, 244, 244, 1)  
test_X = np.repeat(test_X, 3, axis=-1) 
test_ids = test_data['ID'].values  


predictions = model.predict(test_X)
predicted_classes = np.argmax(predictions, axis=1) + 1  


output = pd.DataFrame({'ID': test_ids, 'Predictions': predicted_classes})
output

# # Save to a CSV file
output.to_csv('........../submission_ResNet244_detected(1).csv', index=False)

# print("Predictions saved successfully!")



