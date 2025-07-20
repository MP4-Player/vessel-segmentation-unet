import json
import os
import shutil
from pathlib import Path

# Конфигурация
BASE_DIR = Path("archive/coco2017")
TARGET_CAT_IDS = [44, 46, 47]  # bottle, wine glass, cup

def filter_dataset(annotation_file, image_source_dir, image_target_dir):
    """Фильтрация набора данных"""
    # Создание целевой директории
    image_target_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка аннотаций
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    
    # Сбор ID нужных изображений
    target_image_ids = set()
    for ann in coco_data["annotations"]:
        if ann["category_id"] in TARGET_CAT_IDS:
            target_image_ids.add(ann["image_id"])
    
    # Фильтрация изображений
    filtered_images = []
    for img in coco_data["images"]:
        if img["id"] in target_image_ids:
            filtered_images.append(img)
            src_path = image_source_dir / img["file_name"]
            dst_path = image_target_dir / img["file_name"]
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Ошибка копирования {src_path}: {e}")
    
    # Фильтрация аннотаций
    filtered_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["category_id"] in TARGET_CAT_IDS
    ]
    
    # Фильтрация категорий
    filtered_categories = [
        cat for cat in coco_data["categories"]
        if cat["id"] in TARGET_CAT_IDS
    ]
    
    # Обновление данных
    coco_data["images"] = filtered_images
    coco_data["annotations"] = filtered_annotations
    coco_data["categories"] = filtered_categories
    
    return coco_data

# Обработка тренировочного набора
train_data = filter_dataset(
    annotation_file=BASE_DIR / "annotations/instances_train2017.json",
    image_source_dir=BASE_DIR / "train2017",
    image_target_dir=BASE_DIR / "filtered_train2017"
)
with open(BASE_DIR / "annotations/filtered_train_annotations.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)
print(f"\nТренировочный набор: {len(train_data['images'])} изображений, {len(train_data['annotations'])} аннотаций")

# Обработка валидационного набора
val_data = filter_dataset(
    annotation_file=BASE_DIR / "annotations/instances_val2017.json",
    image_source_dir=BASE_DIR / "val2017",
    image_target_dir=BASE_DIR / "filtered_val2017"
)
with open(BASE_DIR / "annotations/filtered_val_annotations.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)
print(f"Валидационный набор: {len(val_data['images'])} изображений, {len(val_data['annotations'])} аннотаций")

# Для тестового набора (если нужен)
# test_images = [f for f in (BASE_DIR / "test2017").iterdir() if f.is_file()]
# print(f"Тестовый набор: {len(test_images)} изображений (без аннотаций)")