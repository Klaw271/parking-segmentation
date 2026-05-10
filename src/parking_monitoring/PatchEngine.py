import cv2
import numpy as np
from typing import Tuple

class PatchEngine:
    """
    Нарезка изображения на патчи с перекрытием.
    """

    def __init__(self, patch_size: Tuple[int, int], overlap: int):
        self.patch_size = patch_size
        self.overlap = overlap

    def extract(self, img: np.ndarray):

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

