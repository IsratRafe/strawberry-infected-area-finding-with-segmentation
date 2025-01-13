import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.utils import CustomObjectScope

from metrics import dice_coef, dice_loss
from training import load_dataset

# Load the best model
model_path = '/content/data_folder3p/model.h5'
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(model_path)

# Load test data
test_image_dir = '/content/test2/images'
test_mask_dir = '/content/test2/masks'
_, _, (test_x, test_y) = load_dataset('', '', '', '', test_image_dir, test_mask_dir)

# Calculate metrics for the test set
def calculate_metrics(model, dataset_x, dataset_y):
    dice_scores = []
    f1_scores = []
    jaccard_scores = []
    recall_scores = []
    precision_scores = []

    for x, y in tqdm(zip(dataset_x, dataset_y), total=len(dataset_y)):
        # Reading the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # [H, W, 3]
        image = cv2.resize(image, (512, 512))    # [H, W, 3]
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

        # Calculate metrics
        dice = dice_coef(tf.convert_to_tensor(mask, dtype=tf.float32), tf.convert_to_tensor(y_pred, dtype=tf.float32)).numpy()
        f1_value = f1_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary")
        recall_value = recall_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=0)

        # Append metrics
        dice_scores.append(dice)
        f1_scores.append(f1_value)
        jaccard_scores.append(jac_value)
        recall_scores.append(recall_value)
        precision_scores.append(precision_value)

    # Calculate average and standard deviation for all metrics
    metrics = {
        "dice": (np.mean(dice_scores), np.std(dice_scores)),
        "f1": (np.mean(f1_scores), np.std(f1_scores)),
        "jaccard": (np.mean(jaccard_scores), np.std(jaccard_scores)),
        "recall": (np.mean(recall_scores), np.std(recall_scores)),
        "precision": (np.mean(precision_scores), np.std(precision_scores)),
    }

    return metrics

# Calculate metrics for the test set
metrics_test = calculate_metrics(model, test_x, test_y)

# Print the results
print("Test Metrics:")
print(f"Dice Coefficient: {metrics_test['dice'][0]} ± {metrics_test['dice'][1]}")
print(f"F1 Score: {metrics_test['f1'][0]} ± {metrics_test['f1'][1]}")
print(f"Jaccard Index: {metrics_test['jaccard'][0]} ± {metrics_test['jaccard'][1]}")
print(f"Recall: {metrics_test['recall'][0]} ± {metrics_test['recall'][1]}")
print(f"Precision: {metrics_test['precision'][0]} ± {metrics_test['precision'][1]}")
