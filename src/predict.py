import pandas as pd
from src.misc import cm2inch
from dataclasses import dataclass, field
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import torch


matplotlib.rcParams.update({'font.size': 14})


def convert_labelmap_to_color(labelmap):
    lookup_table = np.array([(199, 199, 199), (31, 119, 180), (255, 127, 14)])
    result = np.zeros((*labelmap.shape, 3), dtype=np.uint8)
    np.take(lookup_table, labelmap, axis=0, out=result)
    return result


def plot_confusion_matrix(cm, filename, display_labels=["BG", "S", "W"], xticks_rotation="horizontal",
                          values_format=".1f", figsize=(10, 10)):
    '''
    adapted from https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_plot/confusion_matrix.py#L12
    '''
    cm = cm * 100
    if display_labels is None:
        display_labels = np.arange(cm.shape[0])
    fig, ax = plt.subplots(figsize=cm2inch(figsize))
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], values_format)
        if text_cm == f"{0:.1f}":
            text_cm = "<0.1"
        elif text_cm == f"{100:.1f}":
            text_cm = ">99.9"

        text_[i, j] = ax.text(
            j, i, text_cm,
            ha="center", va="center",
            color=color)
    im_.set_clim(0, 100)
    fig.colorbar(im_, ax=ax)

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True Label",
           xlabel="Predicted Label")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.savefig(f"results/{filename}.png", bbox_inches="tight")
    return


@dataclass
class Predictions:
    filenames: np.ndarray = field(repr=False, init=True)
    img: np.ndarray = field(repr=False, init=True)  # BxCxHxW
    gt: np.ndarray = field(repr=False, init=True)
    model_output: np.ndarray = field(repr=False, init=True)
    pred: np.ndarray = field(repr=False, init=False)
    conf_matrix: np.ndarray = field(repr=False, init=False)
    class_report: pd.DataFrame = field(repr=False, init=False)
    labels = {0: 'background', 1: 'sorghum', 2: 'weed'}
    acc: float = 0

    def __post_init__(self):
        self.n_classes = len(self.labels.keys())
        self.pred = self.model_output
        self.acc, self.class_report, self.conf_matrix = self.calc_pixel_metrics(normalize="true")

    def calc_pixel_metrics(self, normalize=None):
        ann_lbl = self.gt.reshape(-1)
        pred_lbl = self.pred.reshape(-1)
        cm = confusion_matrix(ann_lbl, pred_lbl, labels=list(self.labels.keys()), normalize=normalize)
        cr = classification_report(ann_lbl, pred_lbl, target_names=self.labels.values(),
                                   labels=list(self.labels.keys()), output_dict=True)
        acc, cr_df = self._classification_report_to_df(cr)
        return acc, cr_df, cm

    def _classification_report_to_df(self, cr):
        """
        input: classification report dict from sklearn
        output: accuracy, pandas df without accuracy in it
        """
        acc = cr["accuracy"]
        r = dict(cr)
        del r["accuracy"]
        df = pd.DataFrame(r).T
        return acc, df

    def plot_predictions(self, image_idc=[3, 4, 5, 6]):
        fig, axs = plt.subplots(nrows=len(image_idc), ncols=3, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.01)
        i = np.transpose(self.img, (0, 2, 3, 1))
        p = convert_labelmap_to_color(self.pred)
        g = convert_labelmap_to_color(self.gt)
        # order: image, ground_truth, prediction
        for image_idx, row in zip(image_idc, axs):
            row[0].imshow(i[image_idx])
            row[1].imshow(g[image_idx])
            row[2].imshow(p[image_idx])
        plt.show()
        return

class DiceCalculator:
    def __init__(self, gt, pred, device, n_classes=3):
        self.gt = gt
        self.pred = pred
        self.device = device
        self.n_classes = n_classes
        self.calculate_dice_score()
        self.fix_dice_score()

    def one_hot(self, labelmap, eps=1e-6):
        shp = labelmap.shape
        one_hot = torch.zeros((shp[0], self.n_classes) + shp[1:], device=self.device, dtype=torch.int64)
        return one_hot.scatter_(1, labelmap.unsqueeze(1), 1.0) + eps

    def calculate_dice_score(self, eps: float = 1e-6):
        """
        Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
        """
        input_one_hot = self.one_hot(self.pred)
        target_one_hot: torch.Tensor = self.one_hot(self.gt)
        dims = (2, 3)
        intersection = torch.sum(input_one_hot * target_one_hot, dims)
        cardinality = torch.sum(input_one_hot + target_one_hot, dims)
        dice_score = 2.0 * intersection / (cardinality + eps)
        self.dice_score = dice_score

    def fix_dice_score(self):
        for idx, (gt_img, pred_img) in enumerate(zip(self.gt, self.pred)):
            labels_gt = gt_img.unique()
            labels_pred = pred_img.unique()
            # if there are not 3 labels in the ground truth, we need to change the calculation of the dice_score mean
            if len(labels_gt) == len(labels_pred) and len(labels_gt) == 2:
                if all(torch.eq(labels_gt.to(torch.int8), labels_pred.to(
                        torch.int8))):  # make sure that we only consider matching classes, so the calculation will not change if GT is  only weed and the prediction is only sorghum
                    # get the class that is missing in the ground truth
                    all_labels = torch.tensor([0, 1, 2], device=self.device)
                    # Create a tensor to compare all values at once
                    compareview = labels_gt.repeat(all_labels.shape[0], 1).T
                    # Non Intersection
                    label_idx = all_labels[(compareview != all_labels).T.prod(1) == 1]
                    # replace the values of the dice_score with 'nan' at idx and label_idx cell
                    self.dice_score[idx, label_idx] = torch.nan

            elif len(labels_gt) == len(labels_pred) and len(labels_gt) == 1:  # if there is only background in GT
                if all(torch.eq(labels_gt.to(torch.int8), labels_pred.to(torch.int8))):
                    if labels_gt == 0:
                        self.dice_score[idx, 1] = torch.nan
                        self.dice_score[idx, 2] = torch.nan
        return
