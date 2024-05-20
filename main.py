import os
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR
from tqdm import tqdm, trange

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Networks import Generator, Discriminator, Vocoder
    from Data import VCDataset, VCDataLoader
    from Log import getLogger


if config.Load:
    G = torch.load(config.Gpath, config.device)
    D = torch.load(config.Dpath, config.device)
else:
    G = nn.ModuleDict({
        "XY":   Generator(),
        "YX":   Generator(),
    }).to(config.device)
    D = nn.ModuleDict({
        "X1": Discriminator(),
        "Y1": Discriminator(),
        "X2": Discriminator(),
        "Y2": Discriminator(),
    }).to(config.device)
    
LOSS = nn.ModuleDict({
    "ADV":  nn.BCELoss(),
    "CYC":  nn.L1Loss(),
    "ID":   nn.L1Loss(),
    "ADV2": nn.BCELoss(),
}).to(config.device)
WEIGHT = {
    "ADV":  1,
    "CYC":  10,
    "ID":   5,
    "ADV2": 1,
}
OPTIM = {
    "G": torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999), weight_decay=2e-4),
    "D": torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999), weight_decay=2e-4),
}

trainX = torch.load(os.path.abspath(os.path.join(
    base_folder, config.preprocessed_train_data, "./Ikura/Ikura.data"
)))
trainY = torch.load(os.path.abspath(os.path.join(
    base_folder, config.preprocessed_train_data, "./Trump/Trump.data"
)))
train_dataset = VCDataset(trainX, trainY).to(config.device)
train_dataloader = VCDataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
)
logger = getLogger("MaskCycleGAN-VC", "a" if config.Load else "w")


def train_epoch(epoch):
    G.requires_grad_(False)
    G.eval()
    D.requires_grad_(True)
    D.train
    for (x0, mx, y0, my) in tqdm(train_dataloader, desc="Discriminator Iterations"):
        OPTIM["D"].zero_grad()

        y1 = G["XY"](torch.cat((x0, mx), dim=1))
        x1 = G["YX"](torch.cat((y0, my), dim=1))
        x2 = G["YX"](torch.cat((y1, torch.ones_like(y1)), dim=1))
        y2 = G["XY"](torch.cat((x1, torch.ones_like(x1)), dim=1))

        d1x0, d1x1 = D["X1"](x0), D["X1"](x1)
        d1y0, d1y1 = D["Y1"](y0), D["Y1"](y1)
        d2x0, d2x2 = D["X2"](x0), D["X2"](x2)
        d2y0, d2y2 = D["Y2"](y0), D["Y2"](y2)

        loss = {
            "ADV":  sum((
                LOSS["ADV"](d1x0,  torch.ones_like(d1x0)),
                LOSS["ADV"](d1x1, torch.zeros_like(d1x1)),
                LOSS["ADV"](d1y0,  torch.ones_like(d1y0)),
                LOSS["ADV"](d1y1, torch.zeros_like(d1y1)),
            )),
            "ADV2": sum((
                LOSS["ADV2"](d2x0,  torch.ones_like(d2x0)),
                LOSS["ADV2"](d2x2, torch.zeros_like(d2x2)),
                LOSS["ADV2"](d2y0,  torch.ones_like(d2y0)),
                LOSS["ADV2"](d2y2, torch.zeros_like(d2y2)),
            )),
        }

        loss = sum(tuple(loss[key] * WEIGHT[key] for key in loss.keys()))
        loss.backward()
        OPTIM["D"].step()

    G.requires_grad_(True)
    G.train()
    D.requires_grad_(False)
    D.eval()
    loss_rec = {key: 0 for key in LOSS.keys()}
    for (x0, mx, y0, my) in tqdm(train_dataloader, desc="Generator Iterations"):
        OPTIM["G"].zero_grad()

        y1 = G["XY"](torch.cat((x0, mx), dim=1))
        x1 = G["YX"](torch.cat((y0, my), dim=1))
        x2 = G["YX"](torch.cat((y1, torch.ones_like(y1)), dim=1))
        y2 = G["XY"](torch.cat((x1, torch.ones_like(x1)), dim=1))

        d1x1 = D["X1"](x1)
        d1y1 = D["Y1"](y1)
        d2x2 = D["X2"](x2)
        d2y2 = D["Y2"](y2)

        xid = G["YX"](torch.cat((x0, mx), dim=1))
        yid = G["XY"](torch.cat((y0, my), dim=1))

        loss = {
            "ADV":  sum((
                LOSS["ADV"](d1x1, torch.ones_like(d1x1)),
                LOSS["ADV"](d1y1, torch.ones_like(d1y1)),
            )),
            "CYC":  sum((
                LOSS["CYC"](x2, x0),
                LOSS["CYC"](y2, y0),
            )),
            "ID":   sum((
                LOSS["ID"](xid, x0),
                LOSS["ID"](yid, y0),
            )),
            "ADV2": sum((
                LOSS["ADV2"](d2x2, torch.ones_like(d2x2)),
                LOSS["ADV2"](d2y2, torch.ones_like(d2y2)),
            )),
        }

        for key in loss_rec.keys():
            loss_rec[key] = loss_rec[key] + loss[key].item()

        loss = sum([loss[key] * WEIGHT[key] for key in loss.keys()])
        loss.backward()
        OPTIM["G"].step()

    message = [f"LOSS_{key}: {loss_rec[key]:>10.4f}" for key in loss_rec.keys()]
    message = "\t".join(message)
    logger.info(f"\n[epoch{epoch:0>4}]\n{message}")

    if epoch % 50 == 0:
        torch.save(
            G, os.path.abspath(os.path.join(config.save, f'./G{epoch}.model'))
        )
        torch.save(
            D, os.path.abspath(os.path.join(config.save, f'./D{epoch}.model'))
        )
    pass


if __name__ == '__main__':
    if not os.path.exists(config.save):
        os.mkdir(config.save)

    # for e in trange(1, 51,  desc="Epochs"):
    #     train_epoch(e)

    WEIGHT["ID"] = 0
    for e in trange(4301, 4501, desc="Epochs"):
        train_epoch(e)
    
    pass
