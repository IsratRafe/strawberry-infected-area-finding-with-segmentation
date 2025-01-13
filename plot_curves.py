import pandas as pd
import matplotlib.pyplot as plt

# Define paths to log files
csv_path_local = '/content/data_folder3p/log.csv'
# csv_path_drive = '/content/drive/My Drive/training_logs/log.csv'

# Load the log file
log_local = pd.read_csv(csv_path_local)
# log_drive = pd.read_csv(csv_path_drive)

def plot_metrics(log_file):
    # Adjust the figure size for square plots
    plt.figure(figsize=(8, 8))

    # Plot dice coefficient
    plt.subplot(2, 1, 1)
    plt.plot(log_file['epoch'], log_file['dice_coef'], label='Training Dice Coefficient', color='b')
    plt.plot(log_file['epoch'], log_file['val_dice_coef'], label='Validation Dice Coefficient', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Over Epochs')
    plt.legend()

    # Plot dice loss
    plt.subplot(2, 1, 2)
    plt.plot(log_file['epoch'], log_file['loss'], label='Training Dice Loss', color='b')
    plt.plot(log_file['epoch'], log_file['val_loss'], label='Validation Dice Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('Dice Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot metrics from local log file
plot_metrics(log_local)

# Plot metrics from drive log file
# plot_metrics(log_drive)
