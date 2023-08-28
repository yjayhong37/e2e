# import tensorflow as tf
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os

# # Define the neural network model
# class E2E(tf.keras.Model):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(128, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#         self.output_layer = tf.keras.layers.Dense(3)  # For predicting (x,y,z)

#     def call(self, inputs, **kwargs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         return self.output_layer(x)

# model = E2E()
# optimizer = tf.keras.optimizers.Adam()

# # Initialize lists to store losses for plotting
# train_losses = []

# def load_dataset(data_path, batch_size):
#     # Placeholder function. You need to implement your dataset loading logic here.
#     # This function should yield batches of data.
#     pass

# def train_model(model, dataset_paths, epochs=10):
#     for epoch in range(epochs):
#         for data_path in dataset_paths:
#             # Load your dataset into a format that can be ingested by the model.
#             dataset = load_dataset(data_path, batch_size)  # Remember to define load_dataset()
#             for x_batch, y_batch in dataset:
#                 with tf.GradientTape() as tape:
#                     y_pred = model(x_batch)
#                     loss = tf.keras.losses.MSE(y_batch, y_pred)
                
#                 # Log the error for plotting
#                 train_losses.append(loss.numpy())
                
#                 # Apply gradients
#                 grads = tape.gradient(loss, model.trainable_variables)
#                 optimizer.apply_gradients(zip(grads, model.trainable_variables))

#                 # Optional: Print estimated vs. ground truth for debugging
#                 print(f"Estimated Pose: {y_pred[0].numpy()} Ground Truth: {y_batch[0].numpy()}")

#     # Save the loss plot
#     current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     plt.plot(train_losses)
#     plt.title("Training Loss over Time")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.savefig(os.path.join('logs', f"loss_plot_{current_time}.png"))
#     plt.close()

#     # Save the training log
#     with open(os.path.join('logs', f"training_log_{current_time}.txt"), "w") as log_file:
#         for loss in train_losses:
#             log_file.write(f"{loss}\n")

# # Load your data
# dataset_paths = ['data/3d_oracle_particle.tfrecord']

# # Ensure the 'logs' directory exists
# if not os.path.exists('logs'):
#     os.mkdir('logs')

# # Train your model
# train_model(model, dataset_paths, epochs=10)

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import os
import numpy as np
import datetime

# Define the 3D spatial softmax
def spatial_softmax_3d(feature_map):
    depth, height, width, channels = feature_map.shape[1:]
    feature_map = tf.reshape(feature_map, [-1, depth * height * width, channels])
    softmax = tf.nn.softmax(feature_map)
    softmax = tf.reshape(softmax, [-1, depth, height, width, channels])

    # Compute expected 3D coordinates
    x = tf.linspace(-1., 1., width)
    y = tf.linspace(-1., 1., height)
    z = tf.linspace(-1., 1., depth)
    x, y, z = tf.meshgrid(x, y, z)
    x = tf.reshape(x, [depth * height * width])
    y = tf.reshape(y, [depth * height * width])
    z = tf.reshape(z, [depth * height * width])

    expected_x = tf.reduce_sum(softmax * x, axis=[1, 2, 3])
    expected_y = tf.reduce_sum(softmax * y, axis=[1, 2, 3])
    expected_z = tf.reduce_sum(softmax * z, axis=[1, 2, 3])
    expected_xyz = tf.stack([expected_x, expected_y, expected_z], axis=-1)
    return expected_xyz

# Define the model architecture
# def build_model(input_shape):
#     inputs = layers.Input(input_shape)
#     x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
#     coordinates = spatial_softmax_3d(x)
#     model = Model(inputs, coordinates)
#     return model
def build_model(input_shape):
    inputs = layers.Input(input_shape)
    
    # Layer 1
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    
    # Layer 2
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    
    # Layer 3
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    
    # Layer 4 (Spatial Feature Point Transformation, this is a placeholder. Replace with actual transformation)
    x = layers.Conv2D(128, (1,1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Layer 5
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    
    # Layer 6
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    
    # Layer 7 (Spatial Softmax)
    coordinates = spatial_softmax_3d(x)
    
    model = Model(inputs, coordinates)
    return model


# Load dataset
def load_data(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    images = []
    targets = []
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        target_path = os.path.join(path, image_file.replace('.jpg', '.npy'))
        
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(240, 240))
        image = tf.keras.preprocessing.image.img_to_array(image)
        target = np.load(target_path)
        
        images.append(image)
        targets.append(target)
    
    return np.array(images), np.array(targets)

# Training parameters
learning_rate = 1e-4
batch_size = 32
epochs = 50
input_shape = (240, 240, 3)  # Assuming the images are resized to 240x240

# Load dataset
X, Y = load_data("dataset/")
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(X)).batch(batch_size)

# Model creation
model = build_model(input_shape)
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())

# Model training
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_dataset, epochs=epochs, callbacks=[tensorboard_callback])

print("Model training complete!")

