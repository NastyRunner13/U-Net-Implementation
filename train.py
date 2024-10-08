import torch.amp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNET
import config
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_prediction_as_imgs
)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader)

    for batch_ix, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())


def main():
    
    model = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        config.train_transforms,
        config.val_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )
    
    if config.LOAD_MODEL:
        load_checkpoint(
            torch.load("my_checkpoint.pth.tar", model)
        )

    scaler = torch.amp.grad_scaler('cuda')

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            train_loader, model,
            optimizer, loss_fn, scaler
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        check_accuracy(val_loader, model, device=config.DEVICE)

        save_prediction_as_imgs(val_loader, model, folder="saved_images/", device=config.DEVICE)

if __name__ == "__main__":
    main()