import torch
import random
import numpy as np
#----------------------data augment [1, 80, 80, 80]-------------------------------------------

class RandomFlip:
    def __init__(self):
        pass

    def _flip_lr(self, img):
        """Left-Right flip."""
        return img.flip(3)

    def _flip_ud(self, img):
        """Up-Down flip."""
        return img.flip(2)

    def _flip_fb(self, img):
        """Front-Back flip."""
        return img.flip(1)

    def __call__(self, tem, img, mask):
        """Randomly apply one type of flip (LR, UD, or FB)."""
        flip_type = random.choice(["lr", "ud", "fb"])
        if flip_type == "lr":
            return self._flip_lr(tem), self._flip_lr(img), self._flip_lr(mask)
        elif flip_type == "ud":
            return self._flip_ud(tem), self._flip_ud(img), self._flip_ud(mask)
        elif flip_type == "fb":
            return self._flip_fb(tem), self._flip_fb(img), self._flip_fb(mask)

class RandomRotate:
    def __init__(self):
        pass

    def _rotate_xy(self, img, cnt):
        """Rotate in the XY plane."""
        return torch.rot90(img, cnt, [2, 3])

    def _rotate_xz(self, img, cnt):
        """Rotate in the XZ plane."""
        return torch.rot90(img, cnt, [3, 1])

    def _rotate_yz(self, img, cnt):
        """Rotate in the YZ plane."""
        return torch.rot90(img, cnt, [1, 2])

    def __call__(self, tem, img, mask):
        """Randomly apply one type of rotation (XY, XZ, or YZ)."""
        cnt = random.randint(0, 3)  # Random rotation count (0 to 3)
        rotate_type = random.choice(["xy", "xz", "yz"])
        if rotate_type == "xy":
            return self._rotate_xy(tem, cnt), self._rotate_xy(img, cnt), self._rotate_xy(mask, cnt)
        elif rotate_type == "xz":
            return self._rotate_xz(tem, cnt), self._rotate_xz(img, cnt), self._rotate_xz(mask, cnt)
        elif rotate_type == "yz":
            return self._rotate_yz(tem, cnt), self._rotate_yz(img, cnt), self._rotate_yz(mask, cnt)

class Brightness:
    def __init__(self):
        pass

    def _random_intensity_scale(self, img):
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        img[0, :, :, :] = np.sign(img[0, :, :, :]) * gain * (np.abs(img[0, :, :, :]) ** gamma)
        return img

    def __call__(self, tem, img, mask):
        return tem, self._random_intensity_scale(img), mask  # tem remains unchanged

class ContrastAdjustment:
    def __init__(self):
        pass

    def _adjust_contrast(self, img):
        """
        Adjust the contrast of the image by scaling pixel values around the mean.
        :param img: PyTorch tensor of shape [1, D, H, W]
        :return: Contrast-adjusted tensor
        """
        # Ensure the input is a PyTorch tensor
        assert isinstance(img, torch.Tensor), "Input img must be a PyTorch tensor."
        img = img.clone()  # Avoid modifying the original tensor

        # Generate a random contrast factor in the range [0.8, 1.2]
        contrast_factor = (1.2 - 0.8) * np.random.random_sample() + 0.8

        # Compute the mean of the first channel
        mean = torch.mean(img[0, :, :, :])

        # Adjust contrast: scale values around the mean
        img[0, :, :, :] = contrast_factor * (img[0, :, :, :] - mean) + mean

        return img

    def __call__(self, tem, img, mask):
        """
        Apply contrast adjustment to img only.
        :param tem: PyTorch tensor (temperature map)
        :param img: PyTorch tensor (MRI image)
        :param mask: PyTorch tensor (segmentation mask)
        :return: Adjusted tem, img, and mask
        """
        return tem, self._adjust_contrast(img), mask  # tem and mask remain unchanged

class Compose:
    def __init__(self, transforms):
        """
        初始化Compose，接受一个包含所有增强操作实例的列表。
        """
        self.transforms = transforms  # 传入的是实例化后的对象

    def __call__(self, tem, img, mask):
        """
        有 25% 的概率保留原始数据。
        有 75% 的概率从 transforms 中随机选择一个增强操作执行。
        """
        if random.uniform(0, 1) < 0.25:
            # 保留原始数据
            return tem, img, mask
        else:
            # 遍历 transforms 但只执行一个增强操作
            for t in random.sample(self.transforms, 1):  # 选择并执行一个随机增强操作
                tem, img, mask = t(tem, img, mask)
            return tem, img, mask
