# Скрипт для проведения инференса модели Yolo и создания файла submission.csv
# --output_path путь, по которому проверочная система ожидает результат решения
# --img_dir путь к изображениям из тестового сплита
import os
from pathlib import Path
import pandas as pd
import argparse


def detection_image(image_path, detection_model):
    detections = []
    results = detection_model(image_path)
    results = results[0]
    # Модель ничего не предсказала
    if len(results.boxes.xyxy.cpu()) == 0:
        return ""

    else:
        for cls, xywhn, conf in zip(results.boxes.cls.cpu(), results.boxes.xywhn.cpu(), results.boxes.conf.cpu()):
            detections.append(" ".join([
                str(cls.item()),
                str(xywhn[0].item()),
                str(xywhn[1].item()),
                str(xywhn[2].item()),
                str(xywhn[3].item()),
                str(conf.item())])
            )
        return ";".join(detections)


def inference(detection_model, test_image_dir, output_path):
    results_name = []
    results_detection = []

    for image_name in os.listdir(test_image_dir):
        result_detect = detection_image(
            image_path=os.path.join(test_image_dir, image_name),
            detection_model=detection_model
        )

        results_name.append(Path(image_name).name)
        results_detection.append(result_detect)


    df_result = pd.DataFrame()
    df_result["image_name"] = results_name
    df_result["predicted_detection"] = results_detection

    df_result.to_csv(output_path, index=False)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='images', type=str, help='img dir')
    parser.add_argument('--output_path', default='submission.csv', type=str, help='output path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    path_detection_model_cpt = "best.pt"
    output_path = Path(args.output_path)
    dir_test_images = Path(args.img_dir)

    os.environ['YOLO_CONFIG_DIR'] = os.getcwd() + '/tmp/Ultralytics/configs/'
    from ultralytics import YOLO

    detection_model = YOLO(path_detection_model_cpt)
    inference(detection_model, dir_test_images, output_path)