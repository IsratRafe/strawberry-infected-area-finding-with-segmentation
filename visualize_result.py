import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

from metrics import dice_coef, dice_loss
from training import load_dataset

H = 512
W = 512

model_path = '/content/data_folder3p/model.h5'

def visualize_predictions(model, test_x, test_y, num_samples=3):
    indices = random.sample(range(len(test_x)), num_samples)
    for i in indices:
        x, y = test_x[i], test_y[i]

        # Extracting the name
        name = x.split("/")[-1]

        # Reading the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # [H, W, 3]
        image = cv2.resize(image, (W, H))        # [H, W, 3]
        x = image / 255.0                        # [H, W, 3]
        x = np.expand_dims(x, axis=0)            # [1, H, W, 3]

        # Reading the mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        # Prediction
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(y_pred, cmap='gray')
        plt.axis('off')

        plt.show()

# Load the test data
test_image_dir = '/content/test2/images'
test_mask_dir = '/content/test2/masks'
_, _, (test_x, test_y) = load_dataset('', '', '', '', test_image_dir, test_mask_dir)

# Load the best model
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(model_path)

# Visualize three random sample predictions
visualize_predictions(model, test_x, test_y, num_samples=3)
