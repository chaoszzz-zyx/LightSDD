import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/yolo11-LSCD+PKI.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\NEU-DET.yaml',
                cache=False,
                imgsz=640,#Image input size
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=4,
                # device='0,1'
                optimizer='SGD', # using SGD
                momentum=0.937,#factor of momentum
                weight_decay=0.0005,#Weight attenuation factor
                # patience=0, # set 0 to close earlystop.
                # resume=True
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='LightSDD',
                )