import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from src.misc import seed_all
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io as skio


class SorghumDeBlurDataset(Dataset):
    """
    The dataset class used to compare the deblurring model on its own
    """

    def __init__(self, file_ls, image_folder="data/imgs/"):
        self.images_sharp = None
        self.images_blurry = None
        self.images_deblurred = None
        self.file_ls = file_ls
        self._load_images(file_ls, image_folder)

    def __len__(self):
        return len(self.images_sharp)

    def _load_images(self, file_ls, image_folder: str):
        img_sharp = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        img_blurry = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        img_deblurred = np.zeros(shape=(len(file_ls), 128, 128, 3), dtype=np.uint8)
        for idx, sample_id in enumerate(file_ls):
            image = skio.imread(f"{image_folder}/{sample_id}.png")
            # images are separated by a 4px wide grid
            img_sharp[idx] = image[4:132, 4:132, :3]
            img_blurry[idx] = image[4:132, 136:264, :3]
            img_deblurred[idx] = image[4:132, 400:528, :3]

        self.images_sharp = img_sharp
        self.images_blurry = img_blurry
        self.images_deblurred = img_deblurred
        return

    def __getitem__(self, idx):
        image_sharp = self.images_sharp[idx]
        image_blurry = self.images_blurry[idx]
        image_deblurred = self.images_deblurred[idx]
        return image_sharp, image_blurry, image_deblurred


def normalize_for_lpips(images, factor=255. / 2., cent=1.0):
    images = images.astype('float64')
    norm = images / factor - cent
    return torch.Tensor(norm.transpose((0, 3, 1, 2)))


test = pd.read_csv(f"data/splits/test.csv", dtype=str)
test_ls = list(test["filename"])
b_blurry = True
batch_size = 100
seed = 42
seed_all(seed=seed)
generator = torch.Generator()
generator.manual_seed(seed)
ds = SorghumDeBlurDataset(file_ls=test_ls, image_folder="data/gt_testset")

# sharp - blurry
loss_fn = lpips.LPIPS(net='alex', spatial=False)
img_sharp = normalize_for_lpips(ds.images_sharp)
img_blurry = normalize_for_lpips(ds.images_blurry)
ssim_blurry = ssim(ds.images_sharp, ds.images_blurry, data_range=ds.images_blurry.max() - ds.images_blurry.min(),
                   channel_axis=3)
psnr_blurry = psnr(ds.images_sharp, ds.images_blurry, data_range=ds.images_blurry.max() - ds.images_blurry.min())
lpips_blurry = loss_fn.forward(img_sharp, img_blurry)
print(f"LPIPS sharp-blurry: {lpips_blurry.mean()}")
print(f"SSIM sharp-blurry: {ssim_blurry}")
print(f"PSNR sharp-blurry: {psnr_blurry}")

# sharp - deblurred
loss_fn = lpips.LPIPS(net='alex', spatial=False)
img_sharp = normalize_for_lpips(ds.images_sharp)
img_deblurred = normalize_for_lpips(ds.images_deblurred)
ssim_deblurred = ssim(ds.images_sharp, ds.images_deblurred,
                      data_range=ds.images_deblurred.max() - ds.images_deblurred.min(), channel_axis=3)
psnr_deblurred = psnr(ds.images_sharp, ds.images_deblurred,
                      data_range=ds.images_deblurred.max() - ds.images_deblurred.min())
lpips_deblurred = loss_fn.forward(img_sharp, img_deblurred)
print(f"LPIPS sharp-deblurred: {lpips_deblurred.mean()}")
print(f"SSIM sharp-deblurred: {ssim_deblurred}")
print(f"PSNR sharp-deblurred: {psnr_deblurred}")
