import os
import json

# Конфигурация путей
PROJECT_DIR = r"C:\Users\Пользователь\Desktop\програмироание\БУТЫЛИ"
COCO_DIR = os.path.join(PROJECT_DIR, "archive", "coco2017")
ANNOTATIONS_DIR = os.path.join(COCO_DIR, "annotations")

# Список файлов для проверки
files_to_check = [
    os.path.join(ANNOTATIONS_DIR, "instances_train2017.json"),
    os.path.join(ANNOTATIONS_DIR, "instances_val2017.json"),
    os.path.join(COCO_DIR, "filtered_train_ids.json"),
    os.path.join(COCO_DIR, "filtered_val_ids.json")
]

def print_file_head(file_path, max_chars=500):
    """Печатает начало файла с ограничением по количеству символов"""
    print(f"\n{'='*70}")
    print(f"Файл: {file_path}")
    print(f"Размер: {os.path.getsize(file_path)/1024/1024:.2f} MB")
    print(f"Первые {max_chars} символов:")
    print("-"*50)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_chars)
            print(content)
            
            # Проверка формата
            try:
                json.loads("{" + content.split("{", 1)[-1])
                print("\nФормат: JSON (валидный)")
            except:
                print("\nФормат: Невалидный JSON или бинарный файл")
                
    except Exception as e:
        print(f"Ошибка чтения файла: {str(e)}")
    
    print("="*70)

# Проверяем все файлы
for file_path in files_to_check:
    if os.path.exists(file_path):
        print_file_head(file_path)
    else:
        print(f"\nФайл не существует: {file_path}")