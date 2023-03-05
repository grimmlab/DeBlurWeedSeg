import pandas as pd
import torch
from src.dataset import get_combined_loader, get_comparison_loader
from src.models import DeBlurWeedSeg, WeedSeg
from src.predict import plot_confusion_matrix
from src.misc import seed_all
from pathlib import Path


test = pd.read_csv(f"data/splits/test.csv", dtype=str)
test_ls = list(test["filename"])
seed_all(42)
generator = torch.Generator()
generator.manual_seed(42)
postfix = "combined"
res_path = Path("results")
res_path.mkdir()

weedseg_loader = get_combined_loader(test_ls, image_folder="data/gt_testset", batch_size=200, num_workers=2, split="test", generator=generator, model_type="WeedSeg")
deblurweedseg_loader = get_combined_loader(test_ls, image_folder="data/gt_testset", batch_size=200, num_workers=2, split="test", generator=generator, model_type="DeBlurWeedSeg")

# DeBlurWeedSeg on whole dataset
deblurweedseg = DeBlurWeedSeg("models/model.h5", loader=deblurweedseg_loader, device="cuda")
pred, dsc = deblurweedseg.predict()
# t = (inputs_deblurred.cpu().permute(0,2,3,1)*255.0).int().numpy()  # needed to save the deblurred images
print(f"Dice Score DeBlurWeedSeg {postfix}: {dsc}")
plot_confusion_matrix(pred.conf_matrix, f"DeBlurWeedSeg_{postfix}")

# WeedSeg on whole dataset
weedseg = WeedSeg("models/model.h5", loader=weedseg_loader, device="cuda")
pred, dsc= weedseg.predict()
print(f"Dice Score WeedSeg {postfix}: {dsc}")
plot_confusion_matrix(pred.conf_matrix, f"WeedSeg_{postfix}")


# Both models on sharp/blurry dataset only
for b_blurry in [True, False]:
    if b_blurry == True:
        postfix = "blurry"
    else:
        postfix = "sharp"

    weedseg_loader = get_comparison_loader(test_ls, image_folder="data/gt_testset", batch_size=100, num_workers=2, split="test", b_blurry=b_blurry, generator=generator, model_type="WeedSeg")
    deblurweedseg_loader = get_comparison_loader(test_ls, image_folder="data/gt_testset", batch_size=100, num_workers=2, split="test", b_blurry=b_blurry, generator=generator, model_type="DeBlurWeedSeg")
    deblurweedseg = DeBlurWeedSeg("models/model.h5", loader=deblurweedseg_loader, device="cuda")
    pred, dsc = deblurweedseg.predict()
    # t = (inputs_deblurred.cpu().permute(0,2,3,1)*255.0).int().numpy()  # needed to save the deblurred images
    print(f"Dice Score DeBlurWeedSeg {postfix}: {dsc}")
    plot_confusion_matrix(pred.conf_matrix, f"DeBlurWeedSeg_{postfix}")

    weedseg = WeedSeg("models/model.h5", loader=weedseg_loader, device="cuda")
    pred, dsc= weedseg.predict()
    print(f"Dice Score WeedSeg {postfix}: {dsc}")
    plot_confusion_matrix(pred.conf_matrix, f"WeedSeg_{postfix}")

