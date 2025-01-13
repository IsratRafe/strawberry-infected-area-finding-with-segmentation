import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope

from metrics import dice_coef, dice_loss
from training import load_dataset

# Load the best model
model_path = '/content/data_folder3p/model.h5'
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(model_path)

# Load data
valid_image_dir = '/content/val2/images'
valid_mask_dir = '/content/val2/masks'
test_image_dir = '/content/test2/images'
test_mask_dir = '/content/test2/masks'

_, (valid_x, valid_y), (test_x, test_y) = load_dataset('', '', valid_image_dir, valid_mask_dir, test_image_dir, test_mask_dir)

# Calculate Dice Coefficient for a given dataset
def calculate_dice(model, dataset_x, dataset_y):
    dice_scores = []

    for x, y in tqdm(zip(dataset_x, dataset_y), total=len(dataset_y)):
        # Reading the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # [H, W, 3]
        image = cv2.resize(image, (512, 512))        # [H, W, 3]
        x = image / 255.0                        # [H, W, 3]
        x = np.expand_dims(x, axis=0)            # [1, H, W, 3]

        # Reading the mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))

        # Prediction
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        # Normalize mask to 0 and 1
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.int32)

        # Calculate Dice coefficient
        intersection = np.sum(mask * y_pred)
        dice = (2. * intersection) / (np.sum(mask) + np.sum(y_pred) + 1e-7)

        # Append metric
        dice_scores.append(dice)

    # Calculate average and standard deviation metrics
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    return avg_dice, std_dice

# Calculate metrics for validation and test sets
avg_dice_val, std_dice_val = calculate_dice(model, valid_x, valid_y)
avg_dice_test, std_dice_test = calculate_dice(model, test_x, test_y)

# Print the results
print(f"Validation Dice Coefficient: {avg_dice_val} ± {std_dice_val}")
print(f"Test Dice Coefficient: {avg_dice_test} ± {std_dice_test}")
