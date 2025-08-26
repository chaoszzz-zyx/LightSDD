import os
import shutil
import yaml
from sklearn.model_selection import KFold
from pathlib import Path


dataset_name = "NEU-DET"
yaml_path = r"E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\5Fold\NEU-DET.yaml"
output_root = r"E:\pythonproject\ultralytics-yolo11-20250215\ultralytics-yolo11-main\5Fold"
n_splits = 5



os.makedirs(output_root, exist_ok=True)


with open(yaml_path, encoding='utf-8') as f:
    data = yaml.safe_load(f)


output_root = Path(output_root).resolve()
yaml_dir = Path(yaml_path).parent.resolve()


images_dir = output_root / "images"
labels_dir = output_root / "labels"
folds_dir = output_root / "folds"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)
folds_dir.mkdir(parents=True, exist_ok=True)

all_images = []
for split in ['train', 'val']:
    split_path = data[split].replace('\\', '/')
    img_dir = (yaml_dir / split_path).resolve()

    base_dir = img_dir.parent
    label_dir = base_dir / "labels"

    print(f"处理 {split} 数据集:")
    print(f"  图像目录: {img_dir}")
    print(f"  标签目录: {label_dir}")

    if not img_dir.exists():
        print(f"图像目录不存在: {img_dir}")
        continue

    if not label_dir.exists():
        print(f"标签目录不存在: {label_dir}")
        continue

    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(img_file)

            img_src = img_dir / img_file
            img_dst = images_dir / img_file
            if not img_dst.exists():
                shutil.copy(img_src, img_dst)

            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_src = label_dir / label_file
            label_dst = labels_dir / label_file

            if label_src.exists():
                if not label_dst.exists():
                    shutil.copy(label_src, label_dst)
            else:
                print(f" 找不到标注文件 {label_src}")

if len(all_images) == 0:
    print("没有找到任何图像文件！请检查路径配置。")
    exit(1)

print(f"\n共收集 {len(all_images)} 张图像")

# KFold划分
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    fold_dir = folds_dir / f"fold{fold_idx}"
    fold_dir.mkdir(exist_ok=True)

    images_abs_path = str(images_dir).replace('\\', '/')

    train_list_path = fold_dir / "train.txt"
    val_list_path = fold_dir / "val.txt"

    with open(train_list_path, 'w') as f_train, \
            open(val_list_path, 'w') as f_val:

        for idx in train_idx:
            img_name = all_images[idx]
            f_train.write(f"{images_abs_path}/{img_name}\n")

        for idx in val_idx:
            img_name = all_images[idx]
            f_val.write(f"{images_abs_path}/{img_name}\n")

    fold_config = {
        'train': str(train_list_path).replace('\\', '/'),  
        'val': str(val_list_path).replace('\\', '/'),  
        'nc': data['nc'],
        'names': data['names']
    }

    yaml_path = fold_dir / f"{dataset_name}_fold{fold_idx}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(fold_config, f)

    print(f"生成折叠 {fold_idx} 配置: {yaml_path}")

print(f"\n {n_splits}-fold交叉验证数据集生成完成！")
print(f"项目根目录: {output_root}")

print(f"每折配置: {folds_dir}/foldX/{dataset_name}_foldX.yaml")
