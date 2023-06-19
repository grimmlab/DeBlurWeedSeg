from torch.utils.data import Dataset, DataLoader, ConcatDataset
from albumentations.pytorch import ToTensorV2
import albumentations as a
import numpy as np
from skimage import io as skio
import torch


class SorghumSegmentationDataset(Dataset):
    """
    The dataset class used for segmenting
    """

    def __init__(self,
                 file_ls,  # list of files to add
                 image_folder,  # path to the image folder
                 transform=None,  # which augmentation transforms to use
                 ):
        self.transform = transform
        self.images = None
        self.masks = None
        self.file_ls = file_ls
        self._load_images(file_ls, image_folder)

    def __len__(self):
        return len(self.images)

    def _load_images(self,
                     file_ls: list,  # pandas dataframe loaded from the `label_csv` file
                     image_folder: str):  # path to the image folder
        msk = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        img = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        for idx, sample_id in enumerate(file_ls):
            image = skio.imread(f"{image_folder}/{sample_id}.png")
            # images are separated by a 4px wide grid
            msk_cropped = image[136:264, 4:132, :3]
            msk[idx] = msk_cropped
            img_cropped = image[4:132, 4:132, :3]
            img[idx] = img_cropped

        self.images = img
        self._encode_masks(msk)

        return

    def _encode_masks(self, rgb_mask):
        """
        encodes 4D numpy array
        """
        labels = np.array([[199, 199, 199], [31, 119, 180], [255, 127, 14]], dtype=np.uint8)
        rgb_mask = rgb_mask.reshape((-1, 128, 3))
        label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for idx, label in enumerate(labels):
            label_map[(rgb_mask == np.array(label)).all(axis=2)] = idx
        self.masks = label_map.reshape((-1, 128, 128))
        return

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        fname = self.file_ls[idx]
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return fname, image, mask


def get_loader(file_ls,  # path to the csv containing the labels
               image_folder,  # path to the images
               split: str, # "train", "val", "test" or "final_test"
               generator: torch.Generator,  # Generator to be used for reproducibility
               batch_size: int = 20,  # size of a batch for training and testing
               num_workers: int = 2, # number of workers
               ):
    """
    Loads the dataset as a PyTorch dataloader object for batching
    """
    print(f"Loading split {split}...")
    if split == "train":
        transforms = a.Compose(
            [a.HorizontalFlip(),
             a.VerticalFlip(),
             a.RandomRotate90(),
             a.Transpose(),
             ToTensorV2()])
        shuffle = True

    elif any(substring in split for substring in ['val', 'test']):
        transforms = a.Compose([
            ToTensorV2()])
        shuffle = False
    else:
        raise ValueError(f"Wrong name of labels_csv, please use one of ['train', 'val', 'test', 'final_test']")
    ds = SorghumSegmentationDataset(file_ls=file_ls, image_folder=image_folder, transform=transforms)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, generator=generator)
    return dataloader


class SorghumComparisonDataset(Dataset):
    """
    The dataset class used for comparing WeedSeg with DeBlurWeedSeg
    """
    def __init__(self, file_ls, image_folder="data/imgs/", transform=None, b_blurry=False, model_type="WeedSeg"):
        self.model_type = model_type
        self.transform = transform
        self.images = None
        self.masks = None
        self.file_ls = file_ls
        self.b_blurry = b_blurry
        self._load_images(file_ls, image_folder)

    def __len__(self):
        return len(self.images)

    def _load_images(self, file_ls, image_folder: str):
        msk = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        img = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        for idx, sample_id in enumerate(file_ls):
            image = skio.imread(f"{image_folder}/{sample_id}.png")
            # images are separated by a 4px wide grid
            if self.model_type == "WeedSeg":
                if self.b_blurry:
                    msk_cropped = image[136:264, 136:264, :3]
                    msk[idx] = msk_cropped
                    img_cropped = image[4:132, 136:264, :3]
                    img[idx] = img_cropped
                else:
                    msk_cropped = image[136:264, 4:132, :3]
                    msk[idx] = msk_cropped
                    img_cropped = image[4:132, 4:132, :3]
                    img[idx] = img_cropped
            elif self.model_type == "DeBlurWeedSeg":
                if self.b_blurry:
                    msk_cropped = image[136:264, 400:528, :3]
                    msk[idx] = msk_cropped
                    img_cropped = image[4:132, 136:264, :3] # the image reference will be taken from the non-deblurred part.
                    img[idx] = img_cropped
                else:
                    msk_cropped = image[136:264, 268:396, :3]
                    msk[idx] = msk_cropped
                    img_cropped = image[4:132, 4:132, :3] # the image reference will be taken from the non-deblurred part.
                    img[idx] = img_cropped
            else:
                raise ValueError(f"{self.model_type} not implemented. Use WeedSeg or DeBlurWeedSeg.")
        self.images = img
        self._encode_masks(msk)

        return

    def _encode_masks(self, rgb_mask):
        """
        encodes 4D numpy array
        """
        labels = np.array([[199, 199, 199], [31, 119, 180], [255, 127, 14]], dtype=np.uint8)
        rgb_mask = rgb_mask.reshape((-1, 128, 3))
        label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for idx, label in enumerate(labels):
            label_map[(rgb_mask == np.array(label)).all(axis=2)] = idx
        self.masks = label_map.reshape((-1, 128, 128))
        return

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        fname = self.file_ls[idx]
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return fname, image, mask


def get_comparison_loader(file_ls,  # path to the csv containing the labels
               image_folder,  # path to the images
               split: str, # "train", "val", "test" or "final_test"
               generator: torch.Generator,  # Generator to be used for reproducibility
               batch_size: int = 20,  # size of a batch for training and testing
               num_workers: int = 2, # number of workers
               b_blurry: bool = False,  # whether to load blurry images or masks
               model_type: str = "WeedSeg"):
    """
    Loads the dataset as a PyTorch dataloader object for batching
    """
    print(f"Loading split {split}...")
    if split == "train":
        transforms = a.Compose(
            [a.HorizontalFlip(),
             a.VerticalFlip(),
             a.RandomRotate90(),
             a.Transpose(),
             ToTensorV2()])
        shuffle = True

    elif any(substring in split for substring in ['val', 'test']):
        transforms = a.Compose([
            ToTensorV2()])
        shuffle = False
    else:
        raise ValueError(f"Wrong name of labels_csv, please use one of ['train', 'val', 'test', 'final_test']")
    ds = SorghumComparisonDataset(file_ls, image_folder=image_folder, transform=transforms, b_blurry=b_blurry, model_type=model_type)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, generator=generator)
    return dataloader


def get_combined_loader(file_ls,  # path to the csv containing the labels
               image_folder,  # path to the images
               split: str, # "train", "val", "test" or "final_test"
               generator: torch.Generator,  # Generator to be used for reproducibility
               batch_size: int = 20,  # size of a batch for training and testing
               num_workers: int = 2, # number of workers
               model_type: str = "WeedSeg"):
    """
    Loads the dataset as a PyTorch dataloader object for batching
    """
    print(f"Loading split {split}...")
    if split == "train":
        transforms = a.Compose(
            [a.HorizontalFlip(),
             a.VerticalFlip(),
             a.RandomRotate90(),
             a.Transpose(),
             ToTensorV2()])
        shuffle = True

    elif any(substring in split for substring in ['val', 'test']):
        transforms = a.Compose([
            ToTensorV2()])
        shuffle = False
    else:
        raise ValueError(f"Wrong name of labels_csv, please use one of ['train', 'val', 'test', 'final_test']")
    ds_blurry = SorghumComparisonDataset(file_ls, image_folder=image_folder, transform=transforms, b_blurry=True, model_type=model_type)
    ds_sharp = SorghumComparisonDataset(file_ls, image_folder=image_folder, transform=transforms, b_blurry=False, model_type=model_type)
    ds = ConcatDataset([ds_sharp, ds_blurry])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, generator=generator)
    return dataloader
