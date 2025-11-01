#!/usr/bin/env python3
"""
run_pipeline_pytorch.py

Pipeline completo: prepara datos (masks), stratify split, entrena U-Net (10 epochs),
y realiza inferencia guardando overlays (círculos + etiquetas numeradas) y CSV con coordenadas (px y AU).

Ajusta las rutas y parámetros abajo si es necesario.
"""

import os
import argparse
from glob import glob
import shutil
import csv
import json

import numpy as np
import cv2
from skimage import morphology, measure
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------
# Paths / parámetros
# -------------------------
IMAGES_DIR = "/home/keylor/Documents/snapshot_images_clean"
OUT_BASE = "/home/keylor/Documents/disk_fragment_project/run_output"
PREP_DIR = os.path.join(OUT_BASE, "prepared")   # images + masks + splits
MODEL_DIR = os.path.join(OUT_BASE, "models")
OUT_INFER = os.path.join(OUT_BASE, "inference")
os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_INFER, exist_ok=True)

IMG_EXT = ".png"
IMAGE_SIZE_LIMIT = None  # None => keep original shape, otherwise (H,W) after resize_preserve_aspect

PERCENTILE_MASK = 99.5   # percentil para definir máscara (ajustar si detecta mal)
MIN_AREA_PX = 20         # eliminar objetos más pequeños
CENTER_EXCLUDE_PX = 20   # radio central a excluir en inferencia (px)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dominio físico en AU
L_AU = 1600.0  # usamos [-L_AU, +L_AU] para X e Y si las imágenes están centradas así

# -------------------------
# U-Net
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_filters=32):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base_filters); self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base_filters, base_filters*2); self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(base_filters*2, base_filters*4); self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(base_filters*4, base_filters*8); self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(base_filters*8, base_filters*16)
        self.u6 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 2, stride=2)
        self.c6 = ConvBlock(base_filters*16, base_filters*8)
        self.u7 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.c7 = ConvBlock(base_filters*8, base_filters*4)
        self.u8 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.c8 = ConvBlock(base_filters*4, base_filters*2)
        self.u9 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.c9 = ConvBlock(base_filters*2, base_filters)
        self.out_conv = nn.Conv2d(base_filters, out_ch, 1)
    def forward(self, x):
        def _crop_to(tensor, target_tensor):
            _,_,h,w = target_tensor.size()
            _,_,h_t,w_t = tensor.size()
            dh = h_t - h; dw = w_t - w
            return tensor[:,:,dh//2:dh//2+h, dw//2:dw//2+w]
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))
        c5 = self.c5(self.p4(c4))
        u6 = self.u6(c5); c4c = _crop_to(c4,u6); u6 = torch.cat([u6, c4c], dim=1); c6 = self.c6(u6)
        u7 = self.u7(c6); c3c = _crop_to(c3,u7); u7 = torch.cat([u7, c3c], dim=1); c7 = self.c7(u7)
        u8 = self.u8(c7); c2c = _crop_to(c2,u8); u8 = torch.cat([u8, c2c], dim=1); c8 = self.c8(u8)
        u9 = self.u9(c8); c1c = _crop_to(c1,u9); u9 = torch.cat([u9, c1c], dim=1); c9 = self.c9(u9)
        out = torch.sigmoid(self.out_conv(c9))
        return out

# -------------------------
# Utilidades
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_gray(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    return im

def make_mask_from_image(img, percentile=PERCENTILE_MASK, min_area=MIN_AREA_PX):
    # img: grayscale float (0-255 usually) -> convert to 0-1
    if img.max() > 1.0:
        imf = (img - img.min()) / (img.max() - img.min() + 1e-12)
    else:
        imf = img.copy()
    th = np.percentile(imf.ravel(), percentile)
    bw = imf >= th
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)
    return (bw.astype(np.uint8))

def resize_preserve_aspect(img, target_h, target_w):
    h,w = img.shape
    scale = min(target_h/h, target_w/w)
    new_h = int(h*scale); new_w = int(w*scale)
    im_r = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h,target_w), dtype=img.dtype)
    top = (target_h - new_h)//2; left = (target_w - new_w)//2
    canvas[top:top+new_h, left:left+new_w] = im_r
    return canvas

# -------------------------
# 1) PREPARE dataset: generate masks + stratified splits
# -------------------------
def prepare_dataset(images_dir, out_dir, percent=PERCENTILE_MASK):
    ensure_dir(out_dir)
    img_paths = sorted([os.path.join(images_dir,f) for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXT)])
    print("N imágenes:", len(img_paths))
    imgs = []
    masks = []
    has_fragment = []
    # generate masks from the visual images (grayscale intensity)
    tmp_img_dir = os.path.join(out_dir, "images"); tmp_mask_dir = os.path.join(out_dir, "masks")
    ensure_dir(tmp_img_dir); ensure_dir(tmp_mask_dir)
    for p in tqdm(img_paths, desc="Generando máscaras"):
        im = load_gray(p)
        # opcional: resize to a common size (comment if keep original)
        if IMAGE_SIZE_LIMIT:
            im = resize_preserve_aspect(im, IMAGE_SIZE_LIMIT[0], IMAGE_SIZE_LIMIT[1])
        mask = make_mask_from_image(im, percentile=percent, min_area=MIN_AREA_PX)
        has = int(mask.sum() > 0)
        # save normalized image (0-1) and mask
        im_norm = (im - im.min()) / (im.max() - im.min() + 1e-12)
        save_img_path = os.path.join(tmp_img_dir, os.path.basename(p))
        save_mask_path = os.path.join(tmp_mask_dir, os.path.basename(p).replace(IMG_EXT, "_mask.npy"))
        # save image as npy (C,H,W)
        np.save(save_img_path.replace(".png",".npy"), im_norm.astype(np.float32)[None,...])
        np.save(save_mask_path, mask.astype(np.uint8)[None,...])
        imgs.append(save_img_path.replace(".png",".npy"))
        masks.append(save_mask_path)
        has_fragment.append(has)
    # create df-like lists and stratify by has_fragment
    imgs = np.array(imgs); masks = np.array(masks); has = np.array(has_fragment)
    # train/val/test split stratified on 'has'
    train_imgs, temp_imgs, train_masks, temp_masks, train_has, temp_has = train_test_split(
        imgs, masks, has, test_size=(1-TRAIN_RATIO), stratify=has, random_state=42)
    # split temp into val/test equally
    val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=(1-val_ratio), stratify=temp_has, random_state=42)
    # save lists
    splits_dir = os.path.join(out_dir, "splits"); ensure_dir(splits_dir)
    np.save(os.path.join(splits_dir,"train_imgs.npy"), train_imgs)
    np.save(os.path.join(splits_dir,"train_masks.npy"), train_masks)
    np.save(os.path.join(splits_dir,"val_imgs.npy"), val_imgs)
    np.save(os.path.join(splits_dir,"val_masks.npy"), val_masks)
    np.save(os.path.join(splits_dir,"test_imgs.npy"), test_imgs)
    np.save(os.path.join(splits_dir,"test_masks.npy"), test_masks)
    print("Preparación completada. Splites guardados en:", splits_dir)
    return splits_dir

# -------------------------
# Dataset PyTorch (con padding automático)
# -------------------------
class NpyImageMaskDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def pad_to_multiple(self, tensor, multiple=16):
        """Rellena reflectivamente el tensor hasta que sus dimensiones sean múltiplos de 'multiple'."""
        _, h, w = tensor.shape
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        pad_h = new_h - h
        pad_w = new_w - w
        # F.pad recibe (left, right, top, bottom)
        pad = (0, pad_w, 0, pad_h)
        return F.pad(tensor, pad, mode='reflect')

    def __getitem__(self, idx):
        x = np.load(self.img_paths[idx]).astype(np.float32)  # (1,H,W)
        y = np.load(self.mask_paths[idx]).astype(np.float32) # (1,H,W)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # aplicar padding para que las dimensiones sean múltiplos de 16
        x = self.pad_to_multiple(x)
        y = self.pad_to_multiple(y)

        return x, y


# -------------------------
# Loss + train utils
# -------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    d = dice_loss(pred, target)
    return 0.5*bce + 0.5*d

def train_model(splits_dir, model_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=10):
    train_imgs = np.load(os.path.join(splits_dir, "train_imgs.npy"), allow_pickle=True)
    train_masks = np.load(os.path.join(splits_dir, "train_masks.npy"), allow_pickle=True)
    val_imgs = np.load(os.path.join(splits_dir, "val_imgs.npy"), allow_pickle=True)
    val_masks = np.load(os.path.join(splits_dir, "val_masks.npy"), allow_pickle=True)

    train_ds = NpyImageMaskDataset(train_imgs, train_masks)
    val_ds = NpyImageMaskDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(base_filters=16).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_epoch = 0
    train_losses, val_losses = [], []

    print(f"\n Entrenando hasta {epochs} epochs (early stopping tras {patience} sin mejora)\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward(); optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        train_losses.append(train_loss)

        # === VALIDACIÓN ===
        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x)
                loss = combined_loss(pred, y)
                running_v += loss.item()
        val_loss = running_v / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        # === EARLY STOPPING ===
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, "unet_best.pth"))
            print(f" Mejor modelo actualizado (Val: {val_loss:.5f})")
        elif epoch - best_epoch >= patience:
            print(f" Early stopping: sin mejora en {patience} epochs consecutivas.")
            break

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(model_dir, "unet_last.pth"))

    # === GRAFICAR CURVAS ===
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolución del entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curve.png"))
    plt.close()

    print("\n Entrenamiento completado.")
    print(" Modelos y gráfico guardados en:", model_dir)
    return model


# -------------------------
# Inference + overlays
# -------------------------
def px_to_au(cx, cy, w, h, L=L_AU):
    xs = np.linspace(-L, L, w)
    ys = np.linspace(L, -L, h)
    return float(xs[int(round(cx))]), float(ys[int(round(cy))])

def postprocess_mask(mask, min_area=MIN_AREA_PX):
    # mask is binary uint8
    bw = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)
    return (bw.astype(np.uint8))

def infer_and_save_val(splits_dir, model_path, out_dir, thresh=0.6,
                       center_exclude_px=CENTER_EXCLUDE_PX, min_area_px=MIN_AREA_PX):
    import cv2
    ensure_dir(out_dir)
    # <-- Cambiado de test a val
    val_imgs = np.load(os.path.join(splits_dir, "val_imgs.npy"), allow_pickle=True)
    val_masks = np.load(os.path.join(splits_dir, "val_masks.npy"), allow_pickle=True)

    model = UNet(base_filters=16).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    def pad_to_multiple(tensor, multiple=16):
        dims = tensor.shape
        if len(dims) == 4:
            _, _, h, w = dims
        elif len(dims) == 3:
            _, h, w = dims
        else:
            raise ValueError(f"Forma inesperada del tensor: {dims}")
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        pad_h = new_h - h
        pad_w = new_w - w
        pad = (0, pad_w, 0, pad_h)
        return F.pad(tensor, pad, mode="reflect")

    rows = []
    overlays_dir = os.path.join(out_dir, "overlays_val")
    ensure_dir(overlays_dir)

    for img_path, mask_path in tqdm(zip(val_imgs, val_masks), total=len(val_imgs), desc="Inferencia Val"):
        name = os.path.basename(img_path).replace(".npy", "")
        im = np.load(img_path)[0]  # (H,W)
        H, W = im.shape

        X = torch.from_numpy(im[None, None, ...].astype(np.float32))
        X = pad_to_multiple(X).to(DEVICE)

        with torch.no_grad():
            pred = model(X)[0, 0].cpu().numpy()

        pred = pred[:H, :W]
        pred_bin = (pred > thresh).astype(np.uint8)

        # Componentes conectados
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
        filtered_pred = np.zeros_like(pred_bin)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                filtered_pred[labels == i] = 1
        pred = filtered_pred.astype(np.float32)

        # Suavizado
        pred = cv2.GaussianBlur(pred, (3, 3), 0)

        mask_pred = (pred >= thresh).astype(np.uint8)
        mask_pred = postprocess_mask(mask_pred, min_area=min_area_px)

        yy, xx = np.indices((H, W))
        cy, cx = H // 2, W // 2
        r = np.hypot(xx - cx, yy - cy)
        mask_pred[r <= center_exclude_px] = 0

        labels = measure.label(mask_pred)
        props = measure.regionprops(labels, intensity_image=pred)

        centroids = []
        for j, pobj in enumerate(props, start=1):
            if pobj.area < min_area_px:
                continue
            cyf, cxf = pobj.centroid
            centroids.append((cxf, cyf))
            x_au, y_au = px_to_au(cxf, cyf, W, H)
            rows.append({
                "image": name,
                "fragment_id": j,
                "centroid_px_x": float(cxf),
                "centroid_px_y": float(cyf),
                "centroid_au_x": x_au,
                "centroid_au_y": y_au,
                "area_px": int(pobj.area),
                "max_score": float(pobj.max_intensity)
            })

        # Overlay
        overlay = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        overlay_color = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for idx, (cx, cy) in enumerate(centroids, start=1):
            cv2.circle(overlay_color, (int(round(cx)), int(round(cy))), 12, (0, 0, 255), 2)
            cv2.putText(overlay_color, f"{idx}", (int(round(cx)) + 15, int(round(cy)) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(overlays_dir, f"{name}_overlay.png"), overlay_color)

    # CSV específico para validación
    csv_path = os.path.join(out_dir, "detections_phys_val.csv")
    keys = ["image", "fragment_id", "centroid_px_x", "centroid_px_y",
            "centroid_au_x", "centroid_au_y", "area_px", "max_score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f" Inferencia de validación finalizada. Overlays y CSV guardados en: {out_dir}")

# -------------------------
# MAIN CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline: prepare/train/infer (PyTorch)")
    parser.add_argument("--images_dir", default=IMAGES_DIR)
    parser.add_argument("--prep_dir", default=PREP_DIR)
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--out_infer", default=OUT_INFER)
    parser.add_argument("--do_prepare", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    # Nuevo argumento: umbral de predicción
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.4,
        help="Umbral de probabilidad para generar la máscara binaria en inferencia (default: 0.4)"
    )

    args = parser.parse_args()

    # -------------------------
    # Preparación
    # -------------------------
    if args.do_prepare:
        splits_dir = prepare_dataset(args.images_dir, args.prep_dir)
    else:
        splits_dir = os.path.join(args.prep_dir, "splits")
        if not os.path.exists(splits_dir):
            raise RuntimeError("No se encontraron splits. Ejecuta con --do_prepare primero.")

    # -------------------------
    # Entrenamiento
    # -------------------------
    if args.do_train:
        model = train_model(splits_dir, args.model_dir, epochs=args.epochs)

    # -------------------------
    # Inferencia
    # -------------------------
    if args.do_infer:
        model_path = os.path.join(args.model_dir, "unet_best.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, "unet_last.pth")
        if not os.path.exists(model_path):
            raise RuntimeError("No se encontró modelo en " + args.model_dir)
        
        #  Pasar el umbral al método de inferencia
        infer_and_save_val(splits_dir, model_path, args.out_infer, thresh=args.thresh)



if __name__ == "__main__":
    main()
