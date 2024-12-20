import os

import imgviz
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray


def generate_boxes(img, labels, bboxes, class_names):

    # Создаем подписи для каждого box-а на основе меток классов
    captions = [class_names[label_id] for label_id in labels]

    # Визуализация bounding boxes на изображении с помощью imgviz.instances2rgb
    bbox_viz = imgviz.instances2rgb(
        image=img,                    # Исходное изображение
        bboxes=bboxes,                # Координаты bounding boxes
        labels=labels,                # Метки классов для каждого box-а
        captions=captions,            # Подписи классов
        colormap=imgviz.label_colormap(n_label=10),  # Цветовая карта для классов
        font_size=50,                 # Размер шрифта для подписей
        line_width=10                 # Ширина линии для рамок
    )

    return bbox_viz

def visualize_first_nine(images_dir, predicted_dataframe, MAPPER):
    # Извлекаем путь к изображению по индексу

    plt.figure(figsize=(27, 27))

    for i in range(1, 10):
        image_path = os.path.join(images_dir, predicted_dataframe.loc[i]["image_name"])
        img = Image.open(image_path)

        # Инициализация пустых списков для хранения координат боксов и меток
        bboxes = []
        labels = []

        img_w, img_h = img.size  # Ширина и высота изображения

        # Разбираем предсказания для данного изображения
        for markup in predicted_dataframe.loc[i]["predicted_detection"].split(";"):
            # Обработка предсказания: label, cx, cy, w, h, conf
            label, cx, cy, w, h, conf = markup.split()
            # Преобразуем метку в целое число
            label = int(float(label))
            # Преобразуем координаты в float
            cx, cy, w, h = map(float, [cx, cy, w, h])

            # Предобрабатываем координаты боксов в пикселях
            x1 = int((cx - w/2) * img_w)
            x2 = int((cx + w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            y2 = int((cy + h/2) * img_h)

            # Добавляем метки и боксы в соответствующие списки
            labels.append(label)
            bboxes.append([y1, x1, y2, x2])

        # Визуализируем боксы на изображении
        marked_image = generate_boxes(asarray(img), labels, bboxes, MAPPER)

        plt.subplot(3, 3, i)
        plt.imshow(marked_image)
        plt.axis("off")  # Убираем оси для лучшей видимости изображения

    plt.show()