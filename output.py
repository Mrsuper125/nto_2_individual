import pandas as pd
import os
from pathlib import Path


def detection_image(image_path, detection_model):
    # Инициализация пустого списка для хранения меток и координат ограничивающих рамок
    detections = []
    # Выполняем предсказание на изображении
    results = detection_model(image_path)
    # Получаем результаты (предполагается, что результаты содержат список/кортеж)
    results = results[0]
    # Возвращаем пустую строку, если объектов нет
    if len(results.boxes.xyxy.cpu()) == 0:
        return ""

    else:
        # Для каждого найденного объекта (класс, координаты бокса, уверенность)
        for cls, xywhn, conf in zip(results.boxes.cls.cpu(), results.boxes.xywhn.cpu(), results.boxes.conf.cpu()):
            # Форматируем результаты в строку "class_id x_min y_min x_max y_max confidence"
            detections.append(" ".join([
                str(cls.item()),
                str(xywhn[0].item()),
                str(xywhn[1].item()),
                str(xywhn[2].item()),
                str(xywhn[3].item()),
                str(conf.item())])
            )

        # Объединяем все боксы в одну строку через ";"
        return ";".join(detections)


def inference(detection_model, test_image_dir, output_path) -> pd.DataFrame:
    # Инициализация пустых списков для имен изображений и предсказаний
    results_name = []
    results_detection = []

    # Перебор изображений в директории
    for image_name in os.listdir(test_image_dir):
        image_path = os.path.join(test_image_dir, image_name)

        # Детекция объектов на изображении
        result_detect = detection_image(image_path=image_path, detection_model=detection_model)
        # Добавляем имя изображения в результирующий список
        results_name.append(Path(image_name).name)
        # Добавляем предсказание для изображения в результирующий список
        results_detection.append(result_detect)

    # Создаем DataFrame для результатов
    df_result = pd.DataFrame({
        "image_name": results_name,
        "predicted_detection": results_detection
    })

    return df_result