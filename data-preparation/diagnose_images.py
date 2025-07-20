import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =============================================
# КОНФИГУРАЦИЯ ПУТЕЙ
# =============================================
PROJECT_DIR = r"C:\Users\Пользователь\Desktop\програмироание\БУТЫЛИ"
COCO_DIR = os.path.join(PROJECT_DIR, "archive", "coco2017")
TRAIN_IMAGES_DIR = os.path.join(COCO_DIR, "train2017")
problem_file = "000000297409.jpg"
problem_path = os.path.join(TRAIN_IMAGES_DIR, problem_file)

# =============================================
# ДИАГНОСТИКА ФАЙЛА
# =============================================

def diagnose_image(file_path):
    """Проводит полную диагностику проблемного изображения"""
    print("="*70)
    print(f"ДИАГНОСТИКА ФАЙЛА: {file_path}")
    print("="*70)
    
    # 1. Проверка существования файла
    exists = os.path.exists(file_path)
    print(f"1. Файл существует: {exists}")
    
    if not exists:
        return
    
    # 2. Проверка размера файла
    file_size = os.path.getsize(file_path)
    print(f"2. Размер файла: {file_size} байт")
    
    # 3. Попытка чтения с помощью PIL (библиотека Pillow)
    try:
        pil_img = Image.open(file_path)
        print("3. Успешно открыто с помощью PIL")
        pil_img.verify()  # Проверка целостности
        print("4. Целостность файла подтверждена (PIL verify)")
    except Exception as e:
        print(f"3-4. Ошибка в PIL: {str(e)}")
    
    # 4. Попытка чтения с помощью OpenCV
    try:
        # Стандартный метод
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            print("5. OpenCV.imread: вернул None")
            
            # Альтернативный метод
            with open(file_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), np.uint8)
                cv_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if cv_img is None:
                    print("6. OpenCV.imdecode: вернул None")
                else:
                    print("6. OpenCV.imdecode: успешно!")
                    return cv_img
        else:
            print("5. OpenCV.imread: успешно!")
            return cv_img
    except Exception as e:
        print(f"5-6. Ошибка в OpenCV: {str(e)}")
    
    # 5. Попытка конвертации из PIL в OpenCV
    try:
        pil_img = Image.open(file_path)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print("7. Успешная конвертация PIL -> OpenCV")
        return cv_img
    except Exception as e:
        print(f"7. Ошибка конвертации: {str(e)}")
    
    return None

# =============================================
# ВИЗУАЛИЗАЦИЯ И РЕШЕНИЕ ПРОБЛЕМЫ
# =============================================

def fix_and_visualize(file_path):
    """Решает проблему и показывает изображение"""
    # Запускаем диагностику
    img = diagnose_image(file_path)
    
    print("\nРЕЗУЛЬТАТ ДИАГНОСТИКИ:")
    if img is None:
        print("Не удалось прочитать изображение ни одним методом")
        return
    
    # Показываем изображение
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Успешно прочитано: {os.path.basename(file_path)}")
    plt.axis('off')
    plt.show()
    
    # Сохраняем временную копию для теста
    temp_path = os.path.join(PROJECT_DIR, "temp_fixed.jpg")
    cv2.imwrite(temp_path, img)
    print(f"Создана временная копия: {temp_path}")

# =============================================
# ЗАПУСК ДИАГНОСТИКИ
# =============================================

if __name__ == "__main__":
    fix_and_visualize(problem_path)
    
    # Дополнительная проверка для других файлов
    print("\nПРОВЕРКА ДРУГИХ ФАЙЛОВ:")
    other_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith('.jpg')][:3]
    for file in other_files:
        file_path = os.path.join(TRAIN_IMAGES_DIR, file)
        print(f"\nПроверка: {file}")
        img = diagnose_image(file_path)
        if img is None:
            print(f"⚠️ Ошибка чтения: {file}")
        else:
            print(f"✅ Успешно прочитано: {file}")