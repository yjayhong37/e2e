import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import os
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt


# Define the 2D spatial softmax function
def spatial_softmax_2d(feature_map):
    height, width, channels = feature_map.shape[1:]
    feature_map = tf.reshape(feature_map, [-1, height * width, channels])
    softmax = tf.nn.softmax(feature_map)
    softmax = tf.reshape(softmax, [-1, height, width, channels])

    x_coords = tf.linspace(-1., 1., width)
    y_coords = tf.linspace(-1., 1., height)
    x_coords, y_coords = tf.meshgrid(x_coords, y_coords)

    x_coords = tf.reshape(x_coords, [1, height, width, 1])
    y_coords = tf.reshape(y_coords, [1, height, width, 1])

    expected_x = tf.reduce_sum(x_coords * softmax, axis=[1, 2])
    expected_y = tf.reduce_sum(y_coords * softmax, axis=[1, 2])

    expected_coords = tf.stack([expected_x, expected_y], axis=-1)
    return expected_coords


# Load dataset
def load_data(images_base_dir, labels_base_dir):
    images = []
    targets = []

    vid_dirs = [d for d in os.listdir(images_base_dir) if
                d.startswith('vid_') and os.path.isdir(os.path.join(images_base_dir, d))]

    for vid_dir in vid_dirs:
        # Load images
        image_dir = os.path.join(images_base_dir, vid_dir)
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # Load coordinates from CSV
        vid_number = vid_dir.split('_')[-1]
        csv_name = f'coordinates_{vid_number}.csv'
        csv_path = os.path.join(labels_base_dir, csv_name)

        if not os.path.exists(csv_path):
            print(f"Warning: Missing CSV {csv_path}. Skipping {vid_dir}.")
            continue

        df = pd.read_csv(csv_path)
        label = df.iloc[-1]['x':'y'].values

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)

            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(240, 240))
            image = tf.keras.preprocessing.image.img_to_array(image)

            images.append(image)
            targets.append(label)

    return np.array(images), np.array(targets)


# Training parameters
learning_rate = 1e-5
batch_size = 64
epochs = 100
input_shape = (240, 240, 3)

# Load dataset
IMAGES_BASE_DIR = '/home/yeongjun/DEV/e2e/data/preprocessed_20230830-113324'
LABELS_BASE_DIR = '/home/yeongjun/DEV/e2e/data/2023-08-30_11-10-41/coordinates'
X, Y = load_data(IMAGES_BASE_DIR, LABELS_BASE_DIR)
assert len(X) > 0 and len(Y) > 0, "Datasets are empty!"
X = X.astype('float32') / 255.0
Y = np.expand_dims(Y, axis=1)
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(X)).batch(batch_size)


def build_model(input_shape):
    inputs = layers.Input(input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Flatten the output before feeding into fully connected layers
    x = layers.Flatten()(x)

    # Fully connected layers with dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer for coordinates
    coordinates = layers.Dense(2, activation='linear')(x)

    model = Model(inputs, coordinates)
    return model


# Create and compile the model
model = build_model(input_shape)
model.compile(optimizer=optimizers.AdamW(learning_rate=learning_rate), loss=losses.MeanSquaredError())

# Model training
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
log_file = f"training_logs_{current_date}_{epochs}.txt"
loss_history = []

with open(log_file, "w") as f:
    for epoch in range(epochs):
        history = model.fit(train_dataset, epochs=1)
        loss = history.history['loss'][0]
        loss_history.append(loss)
        f.write(f"Epoch {epoch + 1}/{epochs} - Loss: {loss}\n")
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss}")

# Save loss plot
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
loss_plot_filename = f'loss_plot_{current_date}_{epochs}.png'
plt.savefig(loss_plot_filename)
print(f"Loss plot saved as {loss_plot_filename}!")

print("Model training complete!")

# Close the file
f.close()

