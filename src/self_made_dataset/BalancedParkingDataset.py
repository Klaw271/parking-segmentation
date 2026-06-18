import random
from typing import Any
from torch.utils.data import Dataset


class BalancedParkingDataset(Dataset):
    """
    Комбинированный датасет для балансировки новых и старых данных.

    Объединяет два датасета (старый большой и новый маленький) с возможностью
    задания доли новых данных в каждом epoch. Полезна для fine-tuning моделей
    на новых данных без забывания на старых.

    Attributes:
        old (Dataset): исходный датасет (большой, например 12000 элементов)
        new (Dataset): новый датасет (маленький, например 200 элементов)
        new_ratio (float): доля новых данных (0-1), по умолчанию 0.5
        old_len (int): размер старого датасета
        new_len (int): размер нового датасета
    """

    def __init__(self, old_dataset: Dataset, new_dataset: Dataset, new_ratio: float = 0.5) -> None:
        """
        Инициализирует комбинированный датасет.

        :param old_dataset: исходный датасет (большой)
        :param new_dataset: новый датасет (маленький)
        :param new_ratio: доля новых данных в каждом epoch (по умолчанию 0.5 = 50/50)
                         0.7 = 70% новых, 30% старых (больше акцента на новые данные)
        """

        self.old = old_dataset
        self.new = new_dataset
        self.new_ratio = new_ratio

        self.old_len = len(old_dataset)
        self.new_len = len(new_dataset)

        # создаём индексные списки
        self.old_indices = list(range(self.old_len))
        self.new_indices = list(range(self.new_len))

    def __len__(self) -> int:
        """
        Возвращает размер датасета (размер старого датасета).

        Размер epoch определяется старым датасетом для обработки всех старых данных.

        :return: количество элементов, равное размеру старого датасета
        """
        return self.old_len

    def __getitem__(self, idx: int) -> Any:
        """
        Возвращает случайный элемент из старого или нового датасета.

        С вероятностью new_ratio берёт элемент из нового датасета,
        иначе из старого. Индекс параметра idx используется только для определения
        размера epoch, реальное выбор случайный.

        :param idx: индекс элемента (игнорируется, используется только для размера epoch)
        :return: пара (изображение, маска) из одного из датасетов
        """
        if random.random() < self.new_ratio:
            new_idx = random.randint(0, self.new_len - 1)
            return self.new[new_idx]
        else:
            old_idx = random.randint(0, self.old_len - 1)
            return self.old[old_idx]