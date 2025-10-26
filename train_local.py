import tensorflow as tf
from tensorflow.keras import layers, models

train_dir = '/home/bulipe/ML-practices/ML_datasets/vehicle_data/train'
val_dir   = '/home/bulipe/ML-practices/ML_datasets/vehicle_data/val'

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=8
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=8
)

num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Define a simple CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save model
model.save("vehicle_classifier.keras")
