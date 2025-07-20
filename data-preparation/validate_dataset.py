import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import traceback

# =============================================
# КОНФИГУРАЦИЯ ПУТЕЙ
# =============================================

# Базовый путь к папке с проектом
PROJECT_DIR = r"C:\Users\Пользователь\Desktop\програмироание\БУТЫЛИ"

# Путь к распакованному датасету COCO
COCO_DIR = os.path.join(PROJECT_DIR, "archive", "coco2017")

# Пути к отфильтрованным данным
FILTERED_TRAIN_DIR = os.path.join(COCO_DIR, "filtered_train2017")
FILTERED_VAL_DIR = os.path.join(COCO_DIR, "filtered_val2017")

# Пути к файлам аннотаций
ANNOTATIONS_DIR = os.path.join(COCO_DIR, "annotations")
TRAIN_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "instances_train2017.json")
VAL_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")

# Пути к файлам с ID
TRAIN_IDS_FILE = os.path.join(COCO_DIR, "valid_filtered_train2017_ids.json")
VAL_IDS_FILE = os.path.join(COCO_DIR, "valid_filtered_val2017_ids.json")

# =============================================
# ФУНКЦИИ ДЛЯ ПРОВЕРКИ
# =============================================

def validate_dataset(filtered_dir, annotation_path, ids_file):
    """Проверяет целостность и корректность отфильтрованного датасета"""
    print(f"\nПроверка: {os.path.basename(filtered_dir)}")
    
    # 1. Проверка существования файлов
    if not os.path.exists(filtered_dir):
        print(f"ОШИБКА: Папка не существует - {filtered_dir}")
        return False
    if not os.path.exists(annotation_path):
        print(f"ОШИБКА: Файл аннотаций не найден - {annotation_path}")
        return False
    if not os.path.exists(ids_file):
        print(f"ОШИБКА: Файл с ID не найден - {ids_file}")
        return False
    
    # 2. Загрузка ID изображений
    try:
        with open(ids_file, 'r') as f:
            img_ids = json.load(f)
        print(f"Загружено ID изображений: {len(img_ids)}")
    except Exception as e:
        print(f"ОШИБКА при загрузке {ids_file}: {str(e)}")
        return False
    
    # 3. Проверка количества файлов в папке
    actual_files = os.listdir(filtered_dir)
    print(f"Файлов в папке: {len(actual_files)}")
    
    if len(actual_files) != len(img_ids):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Количество файлов ({len(actual_files)}) не совпадает с количеством ID ({len(img_ids)})")
    
    # 4. Проверка аннотаций
    try:
        print(f"Загрузка аннотаций: {annotation_path}")
        coco = COCO(annotation_path)
        print(f"Аннотации успешно загружены")
        
        # Проверка: есть ли ID в аннотациях
        all_img_ids_in_annot = coco.getImgIds()
        missing_ids = [id for id in img_ids if id not in all_img_ids_in_annot]
        
        if missing_ids:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Найдены ID, отсутствующие в аннотациях: {missing_ids[:10]}... (всего {len(missing_ids)})")
        
        # Проверка случайных изображений
        sample_ids = np.random.choice(img_ids, min(5, len(img_ids)), replace=False)
        print(f"Проверяем случайные изображения: {sample_ids}")
        
        for img_id in sample_ids:
            print(f"\nПроверка изображения ID: {img_id}")
            
            # Исправление: проверяем результат loadImgs перед индексацией
            img_info_list = coco.loadImgs(img_id)
            if not img_info_list:
                print(f"ОШИБКА: Нет информации для изображения с ID={img_id}")
                continue
                
            img_info = img_info_list[0]
            filename = img_info.get('file_name', 'unknown.jpg')
            img_path = os.path.join(filtered_dir, filename)
            
            print(f"Ожидаемый файл: {filename}")
            
            # Проверка существования изображения
            if not os.path.exists(img_path):
                print(f"ОШИБКА: Изображение не найдено - {img_path}")
                continue
            else:
                print(f"Изображение найдено: {img_path}")
                
            # Проверка аннотаций для изображения
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            print(f"Найдено аннотаций: {len(anns)}")
            
            # Проверка, что есть аннотации нужных классов
            target_classes = ["bottle", "cup", "wine glass"]
            cat_ids = coco.getCatIds(catNms=target_classes)
            
            target_anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            print(f"Аннотаций целевых классов: {len(target_anns)}")
            
            if not target_anns:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Для изображения {img_id} нет аннотаций целевых классов")
            
            # Визуализация для проверки
            try:
                img = plt.imread(img_path)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                
                # Показ аннотаций
                for ann in target_anns:
                    if ann.get('segmentation'):
                        try:
                            mask = coco.annToMask(ann)
                            plt.imshow(mask, alpha=0.3, cmap='jet')
                        except Exception as e:
                            print(f"Ошибка создания маски: {str(e)}")
                
                plt.title(f"ID: {img_id} - {filename}\nАннотаций: {len(anns)} | Целевых: {len(target_anns)}")
                plt.axis('off')
                plt.show()
                
            except Exception as e:
                print(f"Ошибка при визуализации изображения {img_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return True
                
    except Exception as e:
        print(f"ОШИБКА при работе с аннотациями: {str(e)}")
        traceback.print_exc()
        return False

# =============================================
# ЗАПУСК ПРОВЕРКИ
# =============================================

print("="*70)
print("НАЧАЛО ПРОВЕРКИ ОТФИЛЬТРОВАННЫХ ДАННЫХ")
print("="*70)

# Проверка тренировочных данных
print("\n" + "="*70)
print("ПРОВЕРКА ТРЕНИРОВОЧНЫХ ДАННЫХ")
print("="*70)
train_valid = validate_dataset(
    filtered_dir=FILTERED_TRAIN_DIR,
    annotation_path=TRAIN_ANNOTATIONS,
    ids_file=TRAIN_IDS_FILE
)

# Проверка валидационных данных
print("\n" + "="*70)
print("ПРОВЕРКА ВАЛИДАЦИОННЫХ ДАННЫХ")
print("="*70)
val_valid = validate_dataset(
    filtered_dir=FILTERED_VAL_DIR,
    annotation_path=VAL_ANNOTATIONS,
    ids_file=VAL_IDS_FILE
)

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
print(f"Тренировочные данные: {'УСПЕШНО' if train_valid else 'С ОШИБКАМИ'}")
print(f"Валидационные данные: {'УСПЕШНО' if val_valid else 'С ОШИБКАМИ'}")
print("="*70)

if train_valid and val_valid:
    print("\n" + "="*70)
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО! ДАННЫЕ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ.")
    print("="*70)
else:
    print("\n" + "="*70)
    print("ОБНАРУЖЕНЫ ПРОБЛЕМЫ. Пожалуйста, проверьте сообщения об ошибках выше.")
    print("="*70)