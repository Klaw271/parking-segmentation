import cv2
import numpy as np


class ImageQualityAnalyzer:
    """
    Класс для оценки качества изображения через анализ границ.
    """

    def __init__(self, threshold_value: int = 200, edge_percent_threshold: float = 1.2):
        self.threshold_value = threshold_value
        self.edge_percent_threshold = edge_percent_threshold

    def analyze(self, image_path: str, visualize: bool = True) -> dict:
        """
        Выполняет анализ качества изображения.

        :param image_path: путь к изображению
        :param visualize: показывать ли визуализацию
        :return: словарь с метриками
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Ошибка загрузки изображения")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])

        filtered = cv2.filter2D(gray, cv2.CV_16S, kernel)
        filtered_abs = cv2.convertScaleAbs(filtered)

        _, edges_binary = cv2.threshold(
            filtered_abs,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY
        )

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

    def _visualize(self, img, filtered, edges):
        import matplotlib.pyplot as plt

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

        return filtered_abs, edges
    
    def visualize_image(self, img, filtered, edges):
        overlay = img.copy()
        overlay[edges > 0] = [0, 0, 255]

        top = np.hstack((img, cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)))
        bottom = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), overlay))

        return np.vstack((top, bottom))