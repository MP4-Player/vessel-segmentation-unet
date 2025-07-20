import os
import shutil
import json
from pycocotools.coco import COCO
import numpy as np

# =============================================
# КОНФИГУРАЦИЯ ПУТЕЙ - ИЗМЕНИТЕ ПО ВАШЕЙ СТРУКТУРЕ
# =============================================
PROJECT_DIR = r"C:\Users\Пользователь\Desktop\програмироание\БУТЫЛИ"
COCO_DIR = os.path.join(PROJECT_DIR, "archive", "coco2017")

# Пути к исходным данным
ANNOTATIONS_DIR = os.path.join(COCO_DIR, "annotations")
TRAIN_IMAGES_DIR = os.path.join(COCO_DIR, "train2017")
VAL_IMAGES_DIR = os.path.join(COCO_DIR, "val2017")

# Пути для сохранения отфильтрованных данных
FILTERED_TRAIN_DIR = os.path.join(COCO_DIR, "filtered_train2017")
FILTERED_VAL_DIR = os.path.join(COCO_DIR, "filtered_val2017")

# Пути к файлам аннотаций
TRAIN_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "instances_train2017.json")
VAL_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")

# =============================================
# ОСНОВНАЯ ФУНКЦИЯ ФИЛЬТРАЦИИ
# =============================================

def filter_coco_dataset(original_dir, target_dir, annotation_path):
    """
    Фильтрует изображения COCO, оставляя только те, что содержат нужные классы
    с усиленной проверкой целостности данных
    """
    # Создаем целевую директорию
    os.makedirs(target_dir, exist_ok=True)
    
    # Загружаем аннотации
    print(f"Загрузка аннотаций: {os.path.basename(annotation_path)}")
    coco = COCO(annotation_path)
    print(f"Аннотации загружены. Всего изображений: {len(coco.imgs)}")
    
    # Классы для сохранения (COCO IDs: bottle=44, cup=47, wine glass=46)
    target_classes = ["bottle", "cup", "wine glass"]
    cat_ids = coco.getCatIds(catNms=target_classes)
    print(f"Целевые классы: {target_classes} (ID: {cat_ids})")
    
    # Получаем ID изображений с нужными классами
    img_ids = set()
    for cat_id in cat_ids:
        ids = coco.getImgIds(catIds=cat_id)
        print(f"Класс ID {cat_id}: найдено {len(ids)} изображений")
        img_ids.update(ids)
    
    print(f"Всего уникальных изображений с целевыми классами: {len(img_ids)}")
    
    # Проверяем каждое изображение перед копированием
    valid_ids = []
    missing_files = []
    invalid_info = []
    
    for img_id in list(img_ids):
        img_info = coco.loadImgs(img_id)
        
        if not img_info:
            print(f"Предупреждение: Нет информации для ID {img_id}")
            invalid_info.append(img_id)
            continue
            
        img_info = img_info[0]
        filename = img_info.get('file_name', '')
        
        if not filename:
            print(f"Предупреждение: Нет имени файла для ID {img_id}")
            invalid_info.append(img_id)
            continue
            
        src_path = os.path.join(original_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"Предупреждение: Файл не найден - {src_path}")
            missing_files.append(src_path)
            continue
            
        valid_ids.append(img_id)
        
        # Копируем изображение
        dst_path = os.path.join(target_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    print("\nРезультат фильтрации:")
    print(f"Успешно скопировано изображений: {len(valid_ids)}")
    print(f"Изображений с отсутствующей информацией: {len(invalid_info)}")
    print(f"Изображений с отсутствующими файлами: {len(missing_files)}")
    
    # Сохраняем только валидные ID
    valid_ids_file = os.path.join(COCO_DIR, f"valid_{os.path.basename(target_dir)}_ids.json")
    with open(valid_ids_file, 'w') as f:
        json.dump(valid_ids, f)
    
    print(f"Список валидных ID сохранен в: {valid_ids_file}")
    return valid_ids

# =============================================
# ЗАПУСК ПРОЦЕССА ФИЛЬТРАЦИИ
# =============================================
def main():
    print("="*70)
    print("ФИЛЬТРАЦИЯ ТРЕНИРОВОЧНЫХ ДАННЫХ")
    print("="*70)
    train_img_ids = filter_coco_dataset(
        original_dir=TRAIN_IMAGES_DIR,
        target_dir=FILTERED_TRAIN_DIR,
        annotation_path=TRAIN_ANNOTATIONS
    )
    
    print("\n" + "="*70)
    print("ФИЛЬТРАЦИЯ ВАЛИДАЦИОННЫХ ДАННЫХ")
    print("="*70)
    val_img_ids = filter_coco_dataset(
        original_dir=VAL_IMAGES_DIR,
        target_dir=FILTERED_VAL_DIR,
        annotation_path=VAL_ANNOTATIONS
    )
    
    print("\n" + "="*70)
    print("ФИЛЬТРАЦИЯ ЗАВЕРШЕНА!")
    print(f"Тренировочные изображения: {len(train_img_ids)}")
    print(f"Валидационные изображения: {len(val_img_ids)}")
    print("="*70)

if __name__ == "__main__":
    main()