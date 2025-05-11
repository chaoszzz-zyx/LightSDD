'''import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
from thop import profile
import pandas as pd
import matplotlib.pyplot as plt
import os

# configuration parameter
plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese
plt.rcParams['axes.unicode_minus'] = False  # Show a negative sign

# Model configuration list
model_configs = [
    {
        'name': 'YOLOv11',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11(300)\results.csv'
    },
    {
        'name': 'YOLOv11-PKI',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11pki(300)\results.csv'
    },
    {
        'name': 'YOLOv11-LSCD',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD(300)\results.csv'
    },
    {
        'name': 'LightSDD',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD+PKI(300)\results.csv'
    }
]


def plot_combined_metrics():
    """Draw the merged mAP curve"""
    try:
        # Create a canvas
        plt.figure(figsize=(14, 6))

        # Draw mAP50
        plt.subplot(1, 2, 1)
        for config in model_configs:
            if os.path.exists(config['csv_path']):
                df = pd.read_csv(config['csv_path'])
                plt.plot(df['epoch'], df['metrics/mAP50(B)'],
                         label=config['name'], linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('mAP50')
        plt.title('mAP50 at IoU=0.5')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        # Draw mAP50-95
        plt.subplot(1, 2, 2)
        for config in model_configs:
            if os.path.exists(config['csv_path']):
                df = pd.read_csv(config['csv_path'])
                plt.plot(df['epoch'], df['metrics/mAP50-95(B)'],
                         label=config['name'], linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('mAP50~95')
        plt.title('mAP for IoU Range 0.5~0.95')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        # Adjust the layout and save
        plt.tight_layout()
        save_dir = 'runs/val/combined_metrics'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/combined_mAP_comparison.png', dpi=300)
        plt.close()
        print(f"The merged comparison chart was successfully saved to {save_dir}")

    except Exception as e:
        print(f"An error occurred when drawing the comparison chart: {str(e)}")


if __name__ == '__main__':
    # Main execution process
    model = YOLO(
        r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD+PKI(300)\weights\best.pt')

    # Verification process
    model.val(
        data=r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\NEU-DET.yaml',
        split='val',
        imgsz=640,
        batch=8,
        project='runs/val',
        name='YOLO11(mAP)',
    )

    # Draw the consolidated indicator chart
    plot_combined_metrics()
'''

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
from thop import profile
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration parameters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese
plt.rcParams['axes.unicode_minus'] = False  # show a negative sign
plt.rcParams['font.size'] = 12  # Unify the font size

# Model configuration list
model_configs = [
    {
        'name': 'YOLOv11',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11(300)\results.csv'
    },
    {
        'name': 'YOLOv11-PKI',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11pki(300)\results.csv'
    },
    {
        'name': 'YOLOv11-LSCD',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD(300)\results.csv'
    },
    {
        'name': 'LightSDD',
        'csv_path': r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD+PKI(300)\results.csv'
    }
]


def plot_loss_curves():
    """Generate a tripartite comparison chart of training and validation losses"""
    try:
        # Create a save directory
        save_dir = 'runs/val/loss_curves'
        os.makedirs(save_dir, exist_ok=True)

        # Training loss tripartite graph
        fig_train, axes_train = plt.subplots(1, 3, figsize=(18, 5))
        train_losses = [
            {'title': 'Training Box Loss', 'col': 'train/box_loss'},
            {'title': 'Training Cls Loss', 'col': 'train/cls_loss'},
            {'title': 'Training DFL Loss', 'col': 'train/dfl_loss'}
        ]

        # Verify the loss tripartite graph
        fig_val, axes_val = plt.subplots(1, 3, figsize=(18, 5))
        val_losses = [
            {'title': 'Validation Box Loss', 'col': 'val/box_loss'},
            {'title': 'Validation Cls Loss', 'col': 'val/cls_loss'},
            {'title': 'Validation DFL Loss', 'col': 'val/dfl_loss'}
        ]

        # Draw the training loss
        for idx, loss in enumerate(train_losses):
            ax = axes_train[idx]
            for config in model_configs:
                if os.path.exists(config['csv_path']):
                    df = pd.read_csv(config['csv_path'])
                    if loss['col'] in df.columns:
                        ax.plot(df['epoch'], df[loss['col']],
                                label=config['name'], linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.set_title(loss['title'], fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=10)

        # Draw the verification loss
        for idx, loss in enumerate(val_losses):
            ax = axes_val[idx]
            for config in model_configs:
                if os.path.exists(config['csv_path']):
                    df = pd.read_csv(config['csv_path'])
                    if loss['col'] in df.columns:
                        ax.plot(df['epoch'], df[loss['col']],
                                label=config['name'], linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.set_title(loss['title'], fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=10)

        # Adjust the layout and save
        fig_train.tight_layout(pad=3.0)
        fig_val.tight_layout(pad=3.0)
        fig_train.savefig(f'{save_dir}/Training_Losses_Comparison.png', dpi=300, bbox_inches='tight')
        fig_val.savefig(f'{save_dir}/Validation_Losses_Comparison.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"The image has been saved to {save_dir}")

    except Exception as e:
        print(f"An error occurred when drawing the loss curve: {str(e)}")


if __name__ == '__main__':
    # Main execution process
    model = YOLO(
        r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\runs\train\YOLO11-LSCD+PKI(300)\weights\best.pt')

    # Verification process
    model.val(data=r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\NEU-DET.yaml',
              split='val',
              imgsz=640,
              batch=8,
              project='runs/val',
              name='YOLO11（loss）')

    # Draw the loss curve
    plot_loss_curves()
