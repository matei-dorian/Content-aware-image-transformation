import random
import torch


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.__is_full():
                p = random.uniform(0, 1)
                if p > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.__add_to_pool(image, idx)
                    return_images.append(tmp)
                else:
                    return_images.append(image)
            else:
                self.__add_to_pool(image)
                return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images

    def __add_to_pool(self, image, replace_idx=None):
        if replace_idx is None:
            self.images.append(image)
            return
        self.images[replace_idx] = image

    def __is_full(self):
        return len(self.images) >= self.pool_size
    
    def state_dict(self):
        return {
            'pool_size': self.pool_size,
            'images': self.images
        }

    def load_state_dict(self, state_dict):
        self.pool_size = state_dict['pool_size']
        self.images = state_dict['images']


