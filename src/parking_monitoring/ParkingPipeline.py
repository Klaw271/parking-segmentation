import os
import cv2
import torch
import matplotlib.pyplot as plt

from src.api.schemas import SuperviselyAnnotation
from src.parking_monitoring.CarDetector import CarDetector
from src.parking_monitoring.DataValidator import DataValidator
from src.parking_monitoring.ImageQualityAnalyzer import ImageQualityAnalyzer
from src.parking_monitoring.OccupancyAnalyzer import OccupancyAnalyzer


class ParkingPipeline:
    """
    Основной pipeline обработки.
    """

    def __init__(self, model_path: str, fail_if_low_quality: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = CarDetector(
            model_path=model_path,
            device=device,
            patch_size=320,
            overlap=160,
            threshold=0.3
        )

        self.quality = ImageQualityAnalyzer()
        self.parking = OccupancyAnalyzer()
        self.validator = DataValidator()
        self.fail_if_low_quality = fail_if_low_quality

    @staticmethod
    def show(img_path, mask, polygons, status):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        overlay = img.copy()
        overlay[mask == 1] = [0, 255, 255]

        left = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        right = img.copy()
        poly_overlay = right.copy()

        for poly, st in zip(polygons, status):
            color = (255, 0, 0) if st else (0, 255, 0)
            cv2.fillPoly(poly_overlay, [poly], color)
            cv2.polylines(right, [poly], True, color, 2)

        right = cv2.addWeighted(poly_overlay, 0.3, right, 0.7, 0)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(left)
        plt.title("Segmentation")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(right)
        plt.title(f"Occupied: {sum(status)}/{len(status)}")
        plt.axis("off")

        plt.show()
    
    def run(self, image_path: str, json_path: str, visualize: bool = True, content_type: str = None, fail_if_low_quality: bool | None = None):
        # 1. Проверка качества
        quality = self.quality.analyze(image_path, visualize=True, content_type=content_type)
        
        print(f"Процент границ: {quality['edge_percent']:.3f}%")
        print("Качество:", "Достаточное" if quality["is_good_quality"] else "Недостаточное")

        strict_check = self.fail_if_low_quality if fail_if_low_quality is None else fail_if_low_quality
        if not quality["is_good_quality"]:
            print("Качество недостаточное.")
            if strict_check:
                print("Статус: Обработка прервана из-за низкого качества")
                return {
                    "status": "error",
                    "message": "Low image qulity",
                    "quality_metrics": quality
                }
            quality["warning"] = "Low quality image, analysis continues"
            print("Статус: Обработка продолжается несмотря на низкое качество")

        # 2. Детекция
        mask = self.detector.detect_patches(image_path, content_type=content_type)

        # 3. Полигональная разметка
        polygons, annotations = self.parking.load_polygons(json_path, SuperviselyAnnotation)
        
        # Вызываем проверку соответствия
        self.validator.verify_consistency(mask, annotations)

        # 4. Анализ
        status = self.parking.check_occupancy(mask, polygons)

        # 5. Отчет
        total = len(polygons)
        occupied = sum(status)
        
        results = {
            "quality": quality,
            "total_spots": total,
            "occupied": occupied,
            "free": total - occupied,
            "occupancy_percent": (occupied / total * 100) if total else 0,
            "status": status,
            "mask": mask, 
            "polygons": polygons
        }

        print("\n=== PARKING REPORT ===")
        print(f"Total: {total}")
        print(f"Occupied: {occupied}")
        print(f"Free: {total - occupied}")
        print(f"Occupancy: {occupied / total * 100 if total else 0:.2f}%")

        # 6. Визуализация
        if visualize:
            self.show(image_path, mask, polygons, status)

        return results


    def get_visualized_image(self, image_path: str, mask, polygons, status) -> bytes:
        """Генерирует PNG картинку с наложенными масками и полигонами."""
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Наложение маски детекции (Cyan)
        overlay_mask = img_rgb.copy()
        overlay_mask[mask == 1] = [0, 255, 255]
        combined = cv2.addWeighted(img_rgb, 0.7, overlay_mask, 0.3, 0)

        # 2. Наложение полигонов (Red/Green)
        poly_overlay = combined.copy()
        for poly, st in zip(polygons, status):
            color = (255, 0, 0) if st else (0, 255, 0)
            cv2.fillPoly(poly_overlay, [poly], color)
            cv2.polylines(combined, [poly], True, color, 2)

        final_img = cv2.addWeighted(poly_overlay, 0.3, combined, 0.7, 0)
        
        # Конвертация в байты
        final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", final_bgr)
        return buffer.tobytes()

# ==============================
# ЗАПУСК
# ==============================
if __name__ == "__main__":
    pipeline = ParkingPipeline("src/models/best_linknet_finetuned.pth", fail_if_low_quality=False)
    test_images_dir = "data/test"
    annotations_dir = os.path.join(test_images_dir, "ann")

    image_files = sorted(
        [f for f in os.listdir(test_images_dir)
         if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(test_images_dir, f))]
    )

    if not image_files:
        print("No JPG images found in data/test.")
    else:
        print(f"Found {len(image_files)} images in {test_images_dir}.")

    ground_truth_occupied = {
        # Задайте эталонные значения занятых мест здесь
        # "filename.jpg": число_занятых_мест,
        "1027208428_0_199_2941_2048_1920x0_80_0_0_ada9346f0d8d23d0df1511f00424463e.jpg": 12,
        "125409101-20150801_170422.jpg": 17,
        "1416729163_505791287.jpg": 6,
        "1493723007159782777.jpg": 8,
        "164984084414064496.jpg": 6,
        "3960b3u-960.jpg": 17,
        "41237.jpg": 25,
        "foto.cheb.ru-085556.jpg": 204,
        "i (1).jpg": 44,
        "i (2).jpg": 66,
        "i.jpg": 14,
        "original.jpg": 49,
        "scale_1200.jpg": 15,
    }

    def evaluate_occupancy(predicted: int, expected: int, total: int) -> dict:
        abs_error = abs(predicted - expected)
        exact_match = predicted == expected
        accuracy_score = 1.0 - abs_error / total if total else (1.0 if exact_match else 0.0)
        if accuracy_score < 0:
            accuracy_score = 0.0
        return {
            "expected": expected,
            "predicted": predicted,
            "absolute_error": abs_error,
            "exact_match": exact_match,
            "count_accuracy": accuracy_score,
        }

    summary = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "compared": 0,
        "exact_matches": 0,
        "sum_absolute_error": 0,
        "sum_total_spots": 0,
        "results": []
    }

    for image_name in image_files:
        image_path = os.path.join(test_images_dir, image_name)
        json_name = f"{image_name}.json"
        json_path = os.path.join(annotations_dir, json_name)

        if not os.path.isfile(json_path):
            print(f"Skipping {image_name}: annotation not found at {json_path}")
            summary["skipped"] += 1
            continue

        print(f"\nProcessing: {image_name}")
        try:
            result = pipeline.run(image_path=image_path, json_path=json_path, visualize=True, fail_if_low_quality=False)
            summary["processed"] += 1

            image_data = {
                "image": image_name,
                "total": result["total_spots"],
                "occupied": result["occupied"],
                "occupancy_percent": result["occupancy_percent"]
            }

            expected = ground_truth_occupied.get(image_name)
            if expected is not None:
                metrics = evaluate_occupancy(result["occupied"], expected, result["total_spots"])
                summary["compared"] += 1
                summary["exact_matches"] += int(metrics["exact_match"])
                summary["sum_absolute_error"] += metrics["absolute_error"]
                summary["sum_total_spots"] += result["total_spots"]
                image_data.update(metrics)
                print(f"Expected occupied: {expected}")
                print(f"Absolute error: {metrics['absolute_error']}")
                print(f"Exact match: {metrics['exact_match']}")
                print(f"Count accuracy: {metrics['count_accuracy'] * 100:.2f}%")
            else:
                print("Ground truth not specified for this image.")

            summary["results"].append(image_data)
        except Exception as exc:
            print(f"Error processing {image_name}: {exc}")
            summary["errors"] += 1

    print("\n=== BATCH SUMMARY ===")
    print(f"Processed: {summary['processed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Errors: {summary['errors']}")
    if summary["compared"]:
        avg_abs_error = summary["sum_absolute_error"] / summary["compared"]
        overall_accuracy = 1.0 - summary["sum_absolute_error"] / summary["sum_total_spots"] if summary["sum_total_spots"] else 0.0
        exact_match_rate = summary["exact_matches"] / summary["compared"]
        print(f"Compared images: {summary['compared']}")
        print(f"Exact match rate: {exact_match_rate * 100:.2f}%")
        print(f"Average absolute error: {avg_abs_error:.2f}")
        print(f"Overall count accuracy: {overall_accuracy * 100:.2f}%")
    else:
        print("No ground truth values were provided for comparison.")