"""B2: ResNet-18 trained on EuroSAT, 12 bands (or RGB with --bands rgb).

The network starts from ImageNet weights (timm "resnet18.a1_in1k"); timm adapts the first convolution to
12 input channels. Each band is normalised with the mean / std of the EuroSAT training split.

Training: AdamW (lr 1e-3, weight decay 1e-4), one warm-up epoch then cosine decay, cross-entropy with
class weights (EuroSAT train has 2,100 forest against 5,600 built-up), random flips and 90-degree
rotations (satellite patches have no "up"). No brightness or colour augmentation, so the domain shift
measured in Part A is not hidden. The epoch with the best validation macro-F1 is kept; training stops
after --patience epochs without improvement. EuroSAT test and the Thai patches are never looked at here.

Writes:
    data/models/resnet18_<bands>.pt        weights plus the bands, classes, normalisation and seed used
    outputs/b/training/resnet18_<bands>_log.csv     loss, validation accuracy and macro-F1 per epoch

Predictions on EuroSAT test and the Thai patches are made by predict.py from the saved checkpoint.

Usage:  python src/train_resnet.py                 (12 bands, 20 epochs)
        python src/train_resnet.py --bands rgb     (B4, B3, B2 only)
        python src/train_resnet.py --seed 1        (repeat with another seed -> resnet18_12band_seed1)
"""
import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import b_data as d
import config as c

BAND_SETS = {"12band": c.BANDS, "rgb": ["B4", "B3", "B2"]}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(n_bands, pretrained=True):
    return timm.create_model("resnet18", pretrained=pretrained, in_chans=n_bands, num_classes=len(c.CLASSES))


def to_input(x_uint16, mean, std, device):
    """uint16 numpy batch (N, bands, 64, 64) -> normalised float32 tensor on the device."""
    x = torch.from_numpy(np.asarray(x_uint16, dtype="float32")).to(device)
    return (x - mean) / std


def augment(x):
    """Random horizontal flip, then a random multiple of 90 degrees, chosen per patch."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    k = torch.randint(0, 4, (len(x),), device=x.device)
    out = x.clone()
    for turns in (1, 2, 3):
        sel = k == turns
        if sel.any():
            out[sel] = torch.rot90(x[sel], turns, dims=(2, 3))
    return out


@torch.no_grad()
def predict_proba(model, x, mean, std, device, batch=512):
    """Softmax probabilities for uint16 patches (N, bands, 64, 64). Used by predict.py as well."""
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        logits = model(to_input(x[start:start + batch], mean, std, device))
        out.append(F.softmax(logits, 1).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", choices=["12band", "rgb"], default="12band")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=c.SEED, help="other seeds are saved as resnet18_<bands>_seed<n>")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    name = f"resnet18_{args.bands}" + ("" if args.seed == c.SEED else f"_seed{args.seed}")
    bands = BAND_SETS[args.bands]
    band_idx = [c.BANDS.index(b) for b in bands]
    print(f"{name}: bands {bands}, device {device}, seed {args.seed}")

    x_all, y_all, split = d.load_eurosat(mmap=False)       # 2.65 GB of uint16 in memory
    x_all = x_all[:, band_idx] if len(band_idx) < len(c.BANDS) else x_all
    train, val = np.flatnonzero(split == d.SPLITS["train"]), np.flatnonzero(split == d.SPLITS["val"])
    x_val, y_val = x_all[val], y_all[val]

    mean_np, std_np = (s[band_idx] for s in d.eurosat_band_stats())
    mean = torch.tensor(mean_np, device=device)[None, :, None, None]
    std = torch.tensor(std_np, device=device)[None, :, None, None]

    counts = np.bincount(y_all[train], minlength=len(c.CLASSES))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    print("train patches per class:", dict(zip(c.CLASSES, counts.tolist())))

    model = build_model(len(bands)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))

    rng = np.random.default_rng(args.seed)
    log, best_f1, best_epoch = [], -1.0, 0
    d.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = d.MODEL_DIR / f"{name}.pt"

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        order = rng.permutation(train)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch):
            ids = np.sort(order[start:start + args.batch])     # sorted indices read memory faster
            x = augment(to_input(x_all[ids], mean, std, device))
            y = torch.from_numpy(y_all[ids]).to(device)
            loss = F.cross_entropy(model(x), y, weight=weights)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            loss_sum += loss.item() * len(ids)

        pred = predict_proba(model, x_val, mean, std, device).argmax(1)
        acc, f1 = accuracy_score(y_val, pred), f1_score(y_val, pred, average="macro")
        log.append({"epoch": epoch, "train_loss": round(loss_sum / len(train), 5), "val_accuracy": round(acc, 5),
                    "val_macro_f1": round(f1, 5), "lr": optimiser.param_groups[0]["lr"], "seconds": round(time.time() - t, 1)})
        mark = ""
        if f1 > best_f1:
            best_f1, best_epoch, mark = f1, epoch, "  (best, saved)"
            torch.save({"model": model.state_dict(), "arch": "resnet18", "bands": bands, "classes": c.CLASSES,
                        "mean": mean_np, "std": std_np, "seed": args.seed, "epoch": epoch, "val_macro_f1": f1}, checkpoint)
        print(f"epoch {epoch:2d}: loss {log[-1]['train_loss']:.4f}, val accuracy {acc:.4f}, "
              f"val macro-F1 {f1:.4f}, {log[-1]['seconds']:.0f} s{mark}", flush=True)
        d.write_table(d.TRAIN_DIR / f"{name}_log.csv", pd.DataFrame(log))
        if epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    print(f"\nbest epoch {best_epoch}, val macro-F1 {best_f1:.4f}; saved {checkpoint}")


if __name__ == "__main__":
    main()
