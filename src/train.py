import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from kornia.losses import DiceLoss
from src.misc import seed_all
from src.early_stopping import EarlyStopping
from src.segmentation_models import UNet
from src.dataset import get_loader, get_combined_loader
from src.predict import DiceCalculator


def train(cfg):
    val_log = []
    max_steps = 5000
    es_patience = 100
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate_every_n_steps = 20
    num_workers = 1
    seed_all(seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    split_path = Path("data/splits")
    gt_path = Path("data/gt")
    early_stop = EarlyStopping(patience=es_patience)
    train = pd.read_csv(f"{split_path}/train.csv", dtype=str)
    train_ls = list(train["filename"])
    val = pd.read_csv(f"{split_path}/val.csv", dtype=str)
    val_ls = list(val["filename"])

    if cfg.training_strategy == "sharp":
        print("Training Strategy: sharp (strategy 2)")
        train_loader = get_loader(train_ls, image_folder=gt_path, batch_size=cfg.batch_size, num_workers=num_workers,
                                  split="train", generator=generator)
        val_loader = get_loader(val_ls, image_folder=gt_path, batch_size=cfg.batch_size, num_workers=num_workers,
                                split="val", generator=generator)
    elif cfg.training_strategy == "combined":
        print("Training Strategy: combined (strategy 1)")
        train_loader = get_combined_loader(train_ls, image_folder=gt_path, batch_size=cfg.batch_size, num_workers=num_workers, split="train",
                            generator=generator, model_type="WeedSeg")
        val_loader = get_combined_loader(val_ls, image_folder=gt_path, batch_size=cfg.batch_size, num_workers=num_workers, split="val",
                            generator=generator, model_type="WeedSeg")
    else:
        raise ValueError(f"No such training strategy called: {cfg.training_strategy}. Try 'sharp' or 'combined'")
    train_iter = iter(train_loader)
    print(f"Training Patches: {len(train_loader.dataset)}")
    print(f"Validation Patches: {len(val_loader.dataset)}")
    loss_fn = DiceLoss()
    scaler = torch.cuda.amp.GradScaler()
    model = UNet(encoder_name=cfg.encoder_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    with tqdm(range(max_steps), unit="batch", leave=False) as tsteps:
        for step in tsteps:
            try:
                _, data_train, target_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                _, data_train, target_train = next(train_iter)

            # train
            data = data_train.float().to(device=device)
            targets = target_train.long().to(device=device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                # backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loss_item = loss.item()
                # log

            # validate
            if step % validate_every_n_steps == 0:
                dice_score = validate_epoch(val_loader, model, device)
                val_log.append([step, dice_score])
                early_stop(dice_score)
                if early_stop.do_stop:
                    print(f"Stopped early at {step}")
                    break
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        val_log = np.asarray(val_log)
        np.savetxt(f"logs/log_{cfg.encoder_name}_{cfg.batch_size}_{cfg.learning_rate}.csv", val_log, delimiter=",",
                   fmt=["%d", "%.6f"])
        return


def retrain(args):
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    gt_path = Path("data/gt")
    split_path = Path("data/splits")
    train = pd.read_csv(f"{str(split_path)}/train.csv", dtype=str)
    train_ls = list(train["filename"])
    val = pd.read_csv(f"{str(split_path)}/val.csv", dtype=str)
    val_ls = list(val["filename"])

    trainval_ls = train_ls + val_ls

    train_loader = get_loader(trainval_ls, image_folder=str(gt_path), batch_size=args.batch_size, num_workers=0,
                              split="train", generator=generator)
    train_iter = iter(train_loader)
    loss_fn = DiceLoss()
    scaler = torch.cuda.amp.GradScaler()
    model = UNet(encoder_name=args.encoder_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    with tqdm(range(args.max_steps+1), unit="batch", leave=False) as tsteps:
        for step in tsteps:
            try:
                _, data_train, target_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                _, data_train, target_train = next(train_iter)

            # train
            data = data_train.float().to(device=device)
            targets = target_train.long().to(device=device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                # backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    print(f"saving model... on step {step}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_name': args.encoder_name,
        'step': step,
    }, f"models/model.h5")
    return

def validate_epoch(loader, model, device):
    dice_scores = []
    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=False) as tepoch:
            for idx, (_, inputs, targets) in enumerate(tepoch):
                inputs = inputs.float().to(device=device)
                targets = targets.long().to(device=device)
                predictions = model(inputs)
                dc = DiceCalculator(targets.cpu().to(torch.int64),
                                    predictions.clone().detach().argmax(axis=1).cpu().to(torch.int64), device="cpu")
                dice_scores.extend(dc.dice_score)
            dice_scores = torch.cat(dice_scores, dim=0)
    model.train()
    return dice_scores.nanmean()
