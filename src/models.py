from NAFNet.predict import Predictor, single_image_inference
from NAFNet.basicsr.utils.options import parse
from NAFNet.basicsr.models import create_model
from src.segmentation_models import UNet
import torch
from src.predict import Predictions, DiceCalculator
import os


class NAFNetPredictor(Predictor):
    def setup(self):
        opt_path_deblur = "options/test/REDS/NAFNet-width64.yml"
        opt_deblur = parse(opt_path_deblur, is_train=False)
        opt_deblur["dist"] = False

        self.model = create_model(opt_deblur)


class WeedSeg:
    def __init__(self, segmentation_model_path, loader, device="cpu"):
        self.seg_model = self.init_seg_model(segmentation_model_path).to(device)
        self.loader = loader
        self.device = device

    def init_seg_model(self, segmentation_model_path):
        loaded_model = torch.load(f"{str(segmentation_model_path)}")
        model = UNet(encoder_name=loaded_model["encoder_name"], pretrained=False)
        print(f"Loading trained weights from {loaded_model['encoder_name']}..., step: {loaded_model['step']}")
        model.load_state_dict(loaded_model["model_state_dict"])
        return model

    def predict(self):
        self.seg_model.eval()
        with torch.no_grad():
            for filename, inputs, targets in self.loader:
                inputs = inputs.float().to(device=self.device)
                targets = targets.to(device=self.device)
                predictions = self.seg_model(inputs)
                dc = DiceCalculator(targets.to(torch.int64), predictions.clone().detach().argmax(axis=1),
                                    device=self.device)
                pred_obj = Predictions(filename, inputs.to(torch.int64).cpu().numpy(),
                                       gt=targets.to(torch.int64).cpu().numpy(),
                                       model_output=predictions.clone().detach().argmax(axis=1).cpu().numpy())
        return pred_obj, dc.dice_score.nanmean()


class DeBlurWeedSeg:
    def __init__(self, segmentation_model_path, loader, device="cpu"):
        self.nafnet = self.init_nafnet()
        self.seg_model = self.init_seg_model(segmentation_model_path).to(device)
        self.loader = loader
        self.device = device

    def init_nafnet(self):
        os.chdir('NAFNet')
        nafnet = NAFNetPredictor()
        nafnet.setup()
        os.chdir('..')
        return nafnet

    def init_seg_model(self, segmentation_model_path):
        loaded_model = torch.load(f"{str(segmentation_model_path)}")
        model = UNet(encoder_name=loaded_model["encoder_name"], pretrained=False)
        print(f"Loading trained weights from {loaded_model['encoder_name']}..., step: {loaded_model['step']}")
        model.load_state_dict(loaded_model["model_state_dict"])
        return model

    def predict(self):
        """
        loader loop needs to be filled with the whole dataset. batch size should equal dataset size
        """
        self.seg_model.eval()
        with torch.no_grad():
            for filename, inputs, targets in self.loader:
                inputs = inputs.float().to(device=self.device)
                targets = targets.to(device=self.device)
                self.nafnet.model.feed_data(data={"lq": inputs / 255.0})
                self.nafnet.model.test()
                inputs_deblurred = self.nafnet.model.get_current_visuals()["result"]
                predictions = self.seg_model((inputs_deblurred * 255.0).to(self.device))
                dc = DiceCalculator(targets.to(torch.int64), predictions.clone().detach().argmax(axis=1),
                                    device=self.device)
                pred_obj = Predictions(filename, inputs_deblurred.to(torch.int64).cpu().numpy(),
                                       gt=targets.to(torch.int64).cpu().numpy(),
                                       model_output=predictions.clone().detach().argmax(axis=1).cpu().numpy())
        return pred_obj, dc.dice_score.nanmean()


