import random
from torch.utils.data import Dataset


class BalancedParkingDataset(Dataset):
    """
    Обёртка над двумя датасетами:
    - old_dataset (12k)
    - new_dataset (200)
    """

    def __init__(self, old_dataset, new_dataset, new_ratio=0.5):
        """
        new_ratio:
            доля новых данных в каждом epoch
            0.5 = 50% new / 50% old
            0.7 = больше акцент на new dataset
        """

        self.old = old_dataset
        self.new = new_dataset
        self.new_ratio = new_ratio

        self.old_len = len(old_dataset)
        self.new_len = len(new_dataset)

        # создаём индексные списки
        self.old_indices = list(range(self.old_len))
        self.new_indices = list(range(self.new_len))

    def __len__(self):
        # делаем epoch размер как у старого датасета
        return self.old_len

    def __getitem__(self, idx):

        # решаем откуда брать данные
        if random.random() < self.new_ratio:
            # NEW dataset (oversampled)
            new_idx = random.randint(0, self.new_len - 1)
            return self.new[new_idx]

        else:
            # OLD dataset
            old_idx = random.randint(0, self.old_len - 1)
            return self.old[old_idx]