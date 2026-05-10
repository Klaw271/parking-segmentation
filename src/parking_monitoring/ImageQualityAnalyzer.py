import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.parking_monitoring.DataValidator import DataValidator


class ImageQualityAnalyzer:
    """
    Класс для оценки качества изображения через анализ границ.
    """

    def __init__(self, threshold_value: int = 200, edge_percent_threshold: float = 1.2):
        self.threshold_value = threshold_value
        self.edge_percent_threshold = edge_percent_threshold
        self.validator = DataValidator()

    def analyze(self, image_path: str, visualize: bool = True, content_type: str = None) -> dict:
        """
        Выполняет анализ качества изображения.

        :param image_path: путь к изображению
        :param visualize: показывать ли визуализацию
        :param content_type: MIME-тип файла для проверки формата
        :return: словарь с метриками
        """
        with open(image_path, 'rb') as f:
            content = f.read()
        
        # Валидатор проверяет размер, формат и целостность
        # и возвращает готовый объект изображения
        img = self.validator.validate_image_source(content, content_type)
        
        filtered_abs, edges_binary, gray = self.process(img)

        edge_pixels = np.count_nonzero(edges_binary)
        total_pixels = gray.size
        edge_percent = (edge_pixels / total_pixels) * 100

        result = {
            "edge_percent": edge_percent,
            "is_good_quality": edge_percent > self.edge_percent_threshold
        }

        if visualize:
            self._visualize(img, filtered_abs, edges_binary)

        return result
        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])

        filtered = cv2.filter2D(gray, cv2.CV_16S, kernel)
        filtered_abs = cv2.convertScaleAbs(filtered)

        _, edges = cv2.threshold(
            filtered_abs,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        return filtered_abs, edges, gray

    def _visualize(self, img, filtered, edges):

        overlay = img.copy()
        overlay[edges > 0] = [0, 0, 255]

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(filtered, cmap='gray')
        plt.title("Filtered (Edges)")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(edges, cmap='gray')
        plt.title("Binary Edges")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        
    def visualize_image(self, img, filtered, edges):
        overlay = img.copy()
        overlay[edges > 0] = [0, 0, 255]

        top = np.hstack((img, cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)))
        bottom = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), overlay))

        return np.vstack((top, bottom))
    
    def get_visualized_report(self, image_path: str, content_type: str = None) -> bytes:
        """
        Полный цикл: расчет и генерация композитного изображения в байтах.
        """
        with open(image_path, 'rb') as f:
            content = f.read()
        
        # Используем валидатор, как и в методе analyze
        img = self.validator.validate_image_source(content, content_type)
        
        # 1. Получаем промежуточные слои
        filtered_abs, edges_binary, _ = self.process(img)
        
        # 2. Формируем сетку 2x2 (ваш метод visualize_image)
        # ВАЖНО: убедитесь, что размеры всех частей совпадают
        overlay = img.copy()
        overlay[edges_binary > 0] = [0, 0, 255] # Красные границы

        # Переводим градации серого в BGR, чтобы hstack сработал
        filtered_bgr = cv2.cvtColor(filtered_abs, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges_binary, cv2.COLOR_GRAY2BGR)

        top = np.hstack((img, filtered_bgr))
        bottom = np.hstack((edges_bgr, overlay))
        result_img = np.vstack((top, bottom))

        # 3. Кодируем в PNG
        _, buffer = cv2.imencode(".png", result_img)
        return buffer.tobytes()