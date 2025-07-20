import json
import os
import shutil
from pathlib import Path

# Настройки с использованием правильных путей
BASE_DIR = Path("archive/coco2017")
ANNOTATION_FILE = BASE_DIR / "annotations/instances_train2017.json"
NEW_ANNOTATION_FILE = BASE_DIR / "annotations/filtered_annotations.json"
IMAGE_SOURCE_DIR = BASE_DIR / "train2017"
IMAGE_TARGET_DIR = BASE_DIR / "filtered_train2017"  # Исправленная опечатка
TARGET_CAT_IDS = [44, 46, 47]  # Бутылки, бокалы, кружки

# Создаем целевую директорию если не существует
IMAGE_TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка аннотаций
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# 1. Собираем ID нужных изображений
target_image_ids = set()
for ann in coco_data["annotations"]:
    if ann["category_id"] in TARGET_CAT_IDS:
        target_image_ids.add(ann["image_id"])

# 2. Фильтруем изображения
filtered_images = []
for img in coco_data["images"]:
    if img["id"] in target_image_ids:
        filtered_images.append(img)
        # Копируем изображение в новую директорию
        src_path = IMAGE_SOURCE_DIR / img["file_name"]
        dst_path = IMAGE_TARGET_DIR / img["file_name"]
        
        # Копирование с обработкой кириллических путей
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Ошибка копирования {src_path}: {e}")

# 3. Фильтруем аннотации
filtered_annotations = [
    ann for ann in coco_data["annotations"]
    if ann["category_id"] in TARGET_CAT_IDS
]

# 4. Обновляем данные COCO
coco_data["images"] = filtered_images
coco_data["annotations"] = filtered_annotations

# 5. Фильтруем категории
coco_data["categories"] = [
    cat for cat in coco_data["categories"]
    if cat["id"] in TARGET_CAT_IDS
]

# Сохраняем новые аннотации
with open(NEW_ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(coco_data, f, indent=2)

print("Фильтрация завершена успешно!")
print(f"Сохранено изображений: {len(filtered_images)}")
print(f"Сохранено аннотаций: {len(filtered_annotations)}")

def verify_annotations(ann_file, image_dir):
    """Проверка целостности отфильтрованных аннотаций"""
    from pycocotools.coco import COCO
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    
    coco = COCO(ann_file)
    print("\nПроверка аннотаций:")
    print(f"Всего изображений: {len(coco.imgs)}")
    print(f"Всего аннотаций: {len(coco.anns)}")
    print(f"Категории: {[coco.cats[cat_id]['name'] for cat_id in coco.cats]}")
    
    # Выбор случайного изображения для визуальной проверки
    img_id = random.choice(list(coco.imgs.keys()))
    img_data = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    print(f"\nПроверка изображения {img_data['file_name']}:")
    print(f"Размер: {img_data['width']}x{img_data['height']}")
    print(f"Найдено аннотаций: {len(anns)}")
    
    # Загрузка и отображение изображения с аннотациями
    img_path = os.path.join(image_dir, img_data['file_name'])
    try:
        img = Image.open(img_path)
        plt.imshow(img)
        coco.showAnns(anns)
        plt.title(f"Image ID: {img_id} | Annotations: {len(anns)}")
        plt.show()
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")

# Проверка тренировочных данных
verify_annotations(NEW_ANNOTATION_FILE, IMAGE_TARGET_DIR)