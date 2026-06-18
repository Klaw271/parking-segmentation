import cv2
import numpy as np
from typing import Tuple, List


class PatchEngine:
    """
    Движок для нарезки изображения на перекрывающиеся патчи.

    Разбивает изображение на квадратные патчи с заданным перекрытием.
    Обрабатывает края изображения путём паддинга нулями и гарантирует
    обработку всей площади, включая углы.

    Attributes:
        patch_size (tuple): размер патча (высота, ширина) в пикселях
        overlap (int): размер перекрытия между соседними патчами в пикселях
    """

    def __init__(self, patch_size: Tuple[int, int], overlap: int) -> None:
        """
        Инициализирует движок нарезки с заданными параметрами.

        :param patch_size: кортеж (высота, ширина) квадратного патча
        :param overlap: размер перекрытия между патчами в пикселях
        """
        self.patch_size = patch_size
        self.overlap = overlap

    def extract(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
        """
        Нарезает изображение на перекрывающиеся патчи.

        Если изображение меньше размера патча, добавляет паддинг нулями.
        Гарантирует обработку углов и полных краёв изображения.

        :param img: входное изображение (H, W) или (H, W, C)
        :return: кортеж (патчи, координаты, размер) где:
                 - патчи: список массивов формата (P_H, P_W) или (P_H, P_W, C)
                 - координаты: список кортежей (y, x) верхних левых углов патчей
                 - размер: кортеж (высота, ширина) обработанного (с паддингом) изображения
        """
        h, w = img.shape[:2]
        p_h, p_w = self.patch_size
        step_h = p_h - self.overlap
        step_w = p_w - self.overlap

        if h < p_h or w < p_w:
            img = cv2.copyMakeBorder(
                img,
                0, max(0, p_h - h),
                0, max(0, p_w - w),
                cv2.BORDER_CONSTANT,
                value=0
            )
            h, w = img.shape[:2]

        patches, coords = [], []

        y_top, y_bottom = 0, h - p_h
        while y_top <= (y_bottom + self.overlap):

            for y in sorted(set([y_top, y_bottom])):
                y = max(0, y)

                x_left, x_right = 0, w - p_w
                while x_left <= (x_right + self.overlap):

                    for x in sorted(set([x_left, x_right])):
                        x = max(0, x)

                        patch = img[y:y+p_h, x:x+p_w]
                        if patch.shape[:2] == (p_h, p_w):
                            patches.append(patch)
                            coords.append((y, x))

                    x_left += step_w
                    x_right -= step_w

            y_top += step_h
            y_bottom -= step_h

        return patches, coords, (h, w)

