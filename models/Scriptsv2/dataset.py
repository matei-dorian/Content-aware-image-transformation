import os

import cv2 as cv
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_anime, root_real, length=None, transforms=None):
        self.root_real = root_real
        self.root_anime = root_anime
        self.transforms = transforms

        self.real_images = os.listdir(self.root_real)
        self.anime_images = os.listdir(self.root_anime)

        self.num_real = len(self.real_images)
        self.num_anime = len(self.anime_images)
        self.length = max(self.num_real, self.num_anime) if length is None else length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_img = self.real_images[idx % self.num_real]
        anime_img = self.anime_images[idx % self.num_anime]

        real_path = self.root_real + "/" + real_img
        anime_path = self.root_anime + "/" + anime_img

        real_img = cv.imread(real_path)
        anime_img = cv.imread(anime_path)

        real_img = cv.cvtColor(real_img, cv.COLOR_BGR2RGB)
        anime_img = cv.cvtColor(anime_img, cv.COLOR_BGR2RGB)

        if self.transforms:
            real_img = self.transforms(real_img)
            anime_img = self.transforms(anime_img)

        return anime_img, real_img
