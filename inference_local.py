import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model("vehicle_classifier.keras")

# Define class names manually if needed
class_names = ['Motorcycles', 'Cars']  # replace with your classes

# Image to predict
# img_path = "/home/bulipe/ML-practices/ML_datasets/vehicle_data/val/Motorcycles/Motorcycle (20).jpg"
img_path = "/home/bulipe/ML-practices/ML_datasets/vehicle_data/val/Cars/Car (24).jpg"

# Preprocess image
img = load_img(img_path, target_size=(224,224))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = img_array / 255.0  # scale

# Predict
pred = model.predict(img_array)
score = tf.nn.softmax(pred[0])
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}%")
