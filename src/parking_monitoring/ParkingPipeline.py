import cv2
import torch
import matplotlib.pyplot as plt

from src.parking_monitoring.CarDetector import CarDetector
from src.parking_monitoring.ImageQualityAnalyzer import ImageQualityAnalyzer
from src.parking_monitoring.OccupancyAnalyzer import OccupancyAnalyzer


class ParkingPipeline:
    """
    Основной pipeline обработки.
    """

    def __init__(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = CarDetector(
            model_path=model_path,
            device=device,
            patch_size=640,
            overlap=320,
            threshold=0.3
        )

        self.quality = ImageQualityAnalyzer()
        self.parking = OccupancyAnalyzer()

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
    
    def run(self, image_path: str, json_path: str):
        # 1. Проверка качества
        quality = self.quality.analyze(image_path)

        # 2. Детекция
        mask = self.detector.detect_patches(image_path)

        # 3. Полигональная разметка
        polygons = self.parking.load_polygons(json_path)

        # 4. Анализ
        status = self.parking.check_occupancy(mask, polygons)

        # 5. Отчет
        total = len(polygons)
        occupied = sum(status)
        
        print(f"Процент границ: {quality['edge_percent']:.3f}%")
        print("Качество:", "Достаточное" if quality["is_good_quality"] else "Недостаточное")

        print("\n=== PARKING REPORT ===")
        print(f"Total: {total}")
        print(f"Occupied: {occupied}")
        print(f"Free: {total - occupied}")
        print(f"Occupancy: {occupied / total * 100:.2f}%")

        # 6. Визуализация
        self.show(image_path, mask, polygons, status)

        return status


# ==============================
# 7. ЗАПУСК
# ==============================
if __name__ == "__main__":
    pipeline = ParkingPipeline("src/models/best_linknet.pth")

    pipeline.run(
        image_path = "data/test/1027208428_0_199_2941_2048_1920x0_80_0_0_ada9346f0d8d23d0df1511f00424463e.jpg",
        json_path = "data/test/ann/1027208428_0_199_2941_2048_1920x0_80_0_0_ada9346f0d8d23d0df1511f00424463e.jpg.json"
    )
    
    
    
''' почти хорошо
image_path = "data/test/3960b3u-960.jpg"
json_path = "data/test/ann/3960b3u-960.jpg.json"
'''
""" ужасно
image_path = "data/test/41237.jpg"
json_path = "data/test/ann/41237.jpg.json"
"""
""" почти хорошо 
image_path = "data/test/1027208428_0_199_2941_2048_1920x0_80_0_0_ada9346f0d8d23d0df1511f00424463e.jpg"
json_path = "data/test/ann/1027208428_0_199_2941_2048_1920x0_80_0_0_ada9346f0d8d23d0df1511f00424463e.jpg.json"
"""
""" хорошо
image_path = "data/test/1416729163_505791287.jpg"
json_path = "data/test/ann/1416729163_505791287.jpg.json"
"""
""" плохо
image_path = "data/test/original.jpg"
json_path = "data/test/ann/original.jpg.json"
"""