"""
=================================================================
TubuleSegmentation v1.0.0 =================================================================
  - EfficientNet-B4 encoder, depth=5, ImageNet weights
  - UnetDecoder (256,128,64,32,16), n_blocks=5, attention='scse'
  - Macenko stain aug (p=0.5), gradient checkpointing encoder+decoders
  - Dice+CE + clDice(0.04) + SCNP(0.16) + Containment(0.24)
  - Warmup topo lineal epochs 20-40
  - batch=4, LR=3e-4, AdamW, CosineAnnealingWarmRestarts

Output: best_model_v1.0.pt
=================================================================
"""
import os, time, numpy as np, cv2, torch, torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from pathlib import Path

# ── Macenko stain augmentation ────────────────────────────────

TORCHSTAIN_OK = False
TorchMacenkoAugmentor = None
try:
    from torchstain.torch.augmentors.macenko import TorchMacenkoAugmentor
    TORCHSTAIN_OK = True
except ImportError:
    try:  # fallback ruta vieja (<1.3.0)
        from torchstain.augmentors.he_augmentor import HEAugmentor as TorchMacenkoAugmentor
        TORCHSTAIN_OK = True
    except ImportError:
        print("  [!] torchstain no encontrado — pip install torchstain")
        print("  [!] Entrenando sin stain augmentation (igual que v21)")

# ============================================================
# TOPOLOGY LOSSES 
# ============================================================
def scnp(logits, labels_onehot, w=3, kappa=1e6):
    fg_mask = labels_onehot.float(); bg_mask = 1.0 - fg_mask
    fg_logits = logits * fg_mask + kappa * bg_mask
    min_pooled = -F.max_pool2d(-fg_logits, kernel_size=w, stride=1, padding=w//2)
    bg_logits = logits * bg_mask - kappa * fg_mask
    max_pooled = F.max_pool2d(bg_logits, kernel_size=w, stride=1, padding=w//2)
    return torch.where(labels_onehot == 1, min_pooled, max_pooled)

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters=3):
    img1 = soft_open(img); skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img); img1 = soft_open(img)
        delta = F.relu(img - img1); skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_loss(pred, target, iters=3, smooth=1e-7):
    skel_pred = soft_skel(pred, iters); skel_target = soft_skel(target, iters)
    tprec = (skel_pred * target).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_target * pred).sum() / (skel_target.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + smooth)

def containment_loss(pred_lumen, pred_epi, smooth=1e-5):
    tissue = torch.clamp(pred_epi + pred_lumen, 0, 1)
    violation = pred_lumen * (1.0 - tissue.detach())
    return violation.sum() / (pred_lumen.sum() + smooth)

# ============================================================
# MODEL v1.0.0
# ============================================================
class TubuleSegModel(nn.Module):
    def __init__(self, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.encoder = smp.encoders.get_encoder(
            'timm-efficientnet-b4', in_channels=3, depth=5, weights='imagenet'
        )
        ec = self.encoder.out_channels
        self.seg_decoder = UnetDecoder(
            encoder_channels=ec, decoder_channels=(256,128,64,32,16),
            n_blocks=5, use_norm='batchnorm', attention_type='scse'
        )
        self.seg_head = nn.Conv2d(16, 3, kernel_size=1)
        self.border_decoder = UnetDecoder(
            encoder_channels=ec, decoder_channels=(256,128,64,32,16),
            n_blocks=5, use_norm='batchnorm', attention_type='scse'
        )
        self.border_head = nn.Conv2d(16, 2, kernel_size=1)

    def _seg_branch(self, *features):
        return self.seg_head(self.seg_decoder(list(features)))

    def _border_branch(self, *features):
        return self.border_head(self.border_decoder(list(features)))

    def _encode(self, x):
        return tuple(self.encoder(x))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            features = list(cp.checkpoint(self._encode, x, use_reentrant=False))
            seg    = cp.checkpoint(self._seg_branch,    *features, use_reentrant=False)
            border = cp.checkpoint(self._border_branch, *features, use_reentrant=False)
            return seg, border
        features = self.encoder(x)
        return (
            self.seg_head(self.seg_decoder(features)),
            self.border_head(self.border_decoder(features))
        )

# ============================================================
# BASE LOSSES
# ============================================================
def multiclass_dice_loss(pred_logits, target_long, smooth=1e-5):
    C = pred_logits.shape[1]; soft = pred_logits.softmax(dim=1)
    oh = F.one_hot(target_long, C).permute(0,3,1,2).float()
    num = 2.0*(soft*oh).sum(dim=(0,2,3))+smooth
    den = soft.sum(dim=(0,2,3))+oh.sum(dim=(0,2,3))+smooth
    return (1.0-(num/den)).mean()

def seg_loss_base(logits, target):
    return multiclass_dice_loss(logits, target) + F.cross_entropy(logits, target)

# ============================================================
# DATA
# ============================================================
def extract_training_data(data_repo, masks_dir):
    orig_dir=data_repo/'tubules_original';conteo_dir=data_repo/'tubules_area_ok'
    images=[];masks=[];borders_outer=[];borders_lumen=[]
    border_pos=np.zeros(2);border_total=0
    n_loaded=0;n_missing=0;_printed_unique=[False]
    def process(orig_path,conteo_path):
        nonlocal border_pos,border_total,n_loaded,n_missing
        orig=cv2.imread(str(orig_path));conteo=cv2.imread(str(conteo_path))
        if orig is None or conteo is None:return
        oh,ow=orig.shape[:2];ch,cw=conteo.shape[:2]
        r=conteo[:,:,2].astype(float);g=conteo[:,:,1].astype(float);b=conteo[:,:,0].astype(float)
        red=((r>150)&(g<100)&(b<100)&((r-g)>80)).astype(np.uint8)
        if ch>oh:red=red[50:ch-50,50:cw-50]
        red=cv2.resize(red,(ow,oh),interpolation=cv2.INTER_NEAREST)
        # ── v23: carga directa de la mascara pre-generada ──────────
        bid=Path(orig_path).stem
        mask_path=masks_dir/f"{bid}.png"
        if not mask_path.exists():
            n_missing+=1;return
        seg_mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if seg_mask is None:
            n_missing+=1;return
        if seg_mask.shape[:2]!=(oh,ow):
            seg_mask=cv2.resize(seg_mask,(ow,oh),interpolation=cv2.INTER_NEAREST)
        seg_mask=seg_mask.astype(np.uint8)
        if not _printed_unique[0]:
            print(f"    [v23] {bid}.png -> labels {np.unique(seg_mask).tolist()} (esperado [0,1,2])")
            _printed_unique[0]=True
        if (seg_mask==1).sum()<oh*ow*0.1:return
        n_loaded+=1
        barrier=cv2.dilate(red,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
        flood=barrier.copy();fm=np.zeros((oh+2,ow+2),dtype=np.uint8)
        for y in range(0,oh,5):
            if flood[y,0]==0:cv2.floodFill(flood,fm,(0,y),128)
            if flood[y,ow-1]==0:cv2.floodFill(flood,fm,(ow-1,y),128)
        for x in range(0,ow,5):
            if flood[0,x]==0:cv2.floodFill(flood,fm,(x,0),128)
            if flood[oh-1,x]==0:cv2.floodFill(flood,fm,(x,oh-1),128)
        bg=(flood==128).astype(np.uint8)
        bg_dil=cv2.dilate(bg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        ob=(red&bg_dil).astype(np.float32);lb=(red&(1-bg_dil)).astype(np.float32)
        k5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        ob=cv2.dilate(ob,k5,iterations=1);lb=cv2.dilate(lb,k5,iterations=1)
        images.append(cv2.resize(orig,(512,512),interpolation=cv2.INTER_LINEAR))
        masks.append(cv2.resize(seg_mask,(512,512),interpolation=cv2.INTER_NEAREST))
        borders_outer.append(cv2.resize(ob,(512,512),interpolation=cv2.INTER_NEAREST))
        borders_lumen.append(cv2.resize(lb,(512,512),interpolation=cv2.INTER_NEAREST))
        border_pos[0]+=borders_outer[-1].sum();border_pos[1]+=borders_lumen[-1].sum()
        border_total+=512*512
    for ap in sorted(conteo_dir.glob('*_conteo.png')):
        bid=get_base_id(ap.name)
        if bid.split('_')[-1]=='396':continue
        op=orig_dir/f"{bid}.tif"
        if not op.exists():continue
        process(op,ap)
    for ext_name in ['tubules_extra','tubules_extra2','tubules_extra4']:
        ext_dir=data_repo/ext_name
        if not ext_dir.exists():continue
        for tp in sorted(ext_dir.glob('*.tif')):
            bid=tp.stem;cp_path=ext_dir/f"{bid}_conteo.png"
            if not cp_path.exists():continue
            process(tp,cp_path)
    pw=np.clip((border_total-border_pos)/(border_pos+1e-6),1.0,15.0) if border_total>0 else np.array([10.,10.])
    print(f"  [v23] mascaras cargadas: {n_loaded} | sin mascara (omitidas): {n_missing}")
    return images,masks,borders_outer,borders_lumen,pw

def get_base_id(f):
    s=os.path.splitext(f)[0]
    for x in['_medida','_conteo','_area']:
        if s.endswith(x):s=s[:-len(x)]
    return s

# ============================================================
# MACENKO STAIN AUGMENTATION — torchstain v1.3.0
# ============================================================
def make_stain_augmentor():
     
    if not TORCHSTAIN_OK:
        return None
    try:
        aug = TorchMacenkoAugmentor(sigma1=0.2, sigma2=0.2)
        return aug
    except Exception as e:
        print(f"  [!] Macenko augmentor init fallo: {e}")
        return None

def apply_stain_aug(img_rgb_uint8, augmentor):
     
    if augmentor is None:
        return img_rgb_uint8
    try:
        # [H,W,3] uint8 -> [3,H,W] uint8 tensor
        img_t = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1)  # uint8, [0,255]
        img_aug, _, _ = augmentor(img_t)   # devuelve (augmented, H, E) tensores
        # [3,H,W] -> [H,W,3] numpy uint8
        img_out = img_aug.permute(1, 2, 0).numpy().astype(np.uint8)
        return img_out
    except Exception:
        return img_rgb_uint8

# ============================================================
# DATASET v1.0.0 — (Macenko stain augmentation)
# ============================================================
class TubuleDataset(Dataset):
    def __init__(self, images, masks, borders_o, borders_l,
                 augment=False, stain_augmentor=None):
        self.images = images
        self.masks = masks
        self.borders_o = borders_o
        self.borders_l = borders_l
        self.augment = augment
        self.stain_augmentor = stain_augmentor

        self.tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
            A.GaussNoise(p=0.3),
        ]) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
        mask = self.masks[idx].copy()
        bo   = self.borders_o[idx].copy()
        bl   = self.borders_l[idx].copy()

        # Macenko stain aug (train, p=0.5)
      
        if self.stain_augmentor is not None and np.random.random() < 0.5:
            img = apply_stain_aug(img, self.stain_augmentor)

        if self.tf:
            bou = (bo * 255).astype(np.uint8)
            blu = (bl * 255).astype(np.uint8)
            t   = self.tf(image=img, masks=[mask, bou, blu])
            img  = t['image']; mask = t['masks'][0]
            bo   = t['masks'][1].astype(np.float32) / 255.0
            bl   = t['masks'][2].astype(np.float32) / 255.0

        img_norm = (img.astype(np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        return (
            torch.from_numpy(img_norm.transpose(2,0,1)).float(),
            torch.from_numpy(mask.copy()).long(),
            torch.from_numpy(np.stack([bo, bl], axis=0)).float()
        )

# ============================================================
# MAIN
# ============================================================
def main():
    base_dir = Path(r"D:\Lu\AI\Tubules"); os.chdir(str(base_dir))
    print("="*60)
    print("  TubuleSegmentation v23")
    print("  EfficientNet-B4 @ 512x512 | batch=4 | SCSE attention")
    print("  Dice+CE + clDice(0.04) + SCNP(0.16) + Containment(0.24)")
    print("  Macenko stain aug (p=0.5) + checkpointing encoder+decoders")
    print(f"  torchstain: {'v1.3.0 OK' if TORCHSTAIN_OK else 'NO (pip install torchstain)'}")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

    from huggingface_hub import snapshot_download
    data_repo = base_dir / 'data_repo'
    if not data_repo.exists():
        snapshot_download('LuGot16/tubules', repo_type='dataset', local_dir=str(data_repo))

    masks_dir = base_dir / 'masks_v2' / 'masks'
    
    print("\n  Loading images + masks_v2...")
    images, masks, borders_o, borders_l, border_pw = extract_training_data(data_repo, masks_dir)
    print(f"  Total images: {len(images)}")
    print(f"  Border pos_weight: outer={border_pw[0]:.1f}, lumen={border_pw[1]:.1f}")

    np.random.seed(42)
    idx = np.random.permutation(len(images))
    sp  = int(len(idx) * 0.85)
    train_idx = idx[:sp]; val_idx = idx[sp:]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    stain_aug = make_stain_augmentor()
    if stain_aug is not None:
        print("  Macenko stain augmentation: YES (sigma=0.2, p=0.5 por imagen)")
    else:
        print("  Macenko stain augmentation: NO")

    train_ds = TubuleDataset(
        [images[i] for i in train_idx], [masks[i] for i in train_idx],
        [borders_o[i] for i in train_idx], [borders_l[i] for i in train_idx],
        augment=True, stain_augmentor=stain_aug
    )
    val_ds = TubuleDataset(
        [images[i] for i in val_idx], [masks[i] for i in val_idx],
        [borders_o[i] for i in val_idx], [borders_l[i] for i in val_idx],
        augment=False, stain_augmentor=None
    )

    BATCH = 4
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    model = TubuleSegModel(use_checkpoint=True); model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  Model: {n_params:.1f}M params (SCSE attention en decoders)")

    # ---- SMOKE TEST ----
    model.train()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    _x = torch.randn(2, 3, 512, 512, device=device)
    _seg, _bor = model(_x)
    assert _seg.shape == (2,3,512,512) and _bor.shape == (2,2,512,512)
    (_seg.mean() + _bor.mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), "grad no finito"
    model.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        print(f"  Smoke test OK | VRAM_peak smoke={torch.cuda.max_memory_allocated()/1024**2:.0f}MB (batch=2)")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("  Smoke test OK")

    W_CLDICE = 0.04; W_SCNP = 0.16; W_CONTAIN = 0.24
    TOPO_START = 20; TOPO_END = 40
    EPOCHS = 200; PATIENCE = 80; LR = 3e-4

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
    border_pw_t = torch.tensor(border_pw, dtype=torch.float32).to(device)

    best = 0; pat = 0
    model_path = base_dir / 'best_model_v23.pt'

    print(f"\n  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print(f"  Topo: clDice={W_CLDICE}, SCNP={W_SCNP}, Contain={W_CONTAIN}")
    print(f"  Warmup: epochs {TOPO_START}-{TOPO_END} (linear 0->1) | Batch: {BATCH}")
    print(f"{'='*60}")
    t0 = time.time()

    for ep in range(EPOCHS):
        if ep < TOPO_START:   topo_scale = 0.0
        elif ep < TOPO_END:   topo_scale = (ep - TOPO_START) / (TOPO_END - TOPO_START)
        else:                 topo_scale = 1.0

        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        model.train(); tl = 0; tb = 0

        for imgs, seg_masks, borders in train_dl:
            imgs      = imgs.to(device)
            seg_masks = seg_masks.to(device)
            borders   = borders.to(device)
            opt.zero_grad()

            seg_logits, border_logits = model(imgs)
            l_base   = seg_loss_base(seg_logits, seg_masks)
            l_border = F.binary_cross_entropy_with_logits(
                border_logits.float(), borders.float(),
                pos_weight=border_pw_t.view(1,2,1,1))
            loss = l_base + 0.2 * l_border

            if topo_scale > 0:
                seg_probs  = seg_logits.softmax(dim=1)
                epi_pred   = seg_probs[:,1:2,:,:]
                epi_target = (seg_masks==1).float().unsqueeze(1)
                l_cldice   = cldice_loss(epi_pred, epi_target, iters=3)
                labels_oh  = F.one_hot(seg_masks, 3).permute(0,3,1,2).float()
                z_tilde    = scnp(seg_logits, labels_oh, w=3)
                l_scnp     = F.cross_entropy(z_tilde, seg_masks)
                l_contain  = containment_loss(seg_probs[:,2,:,:], seg_probs[:,1,:,:])
                loss = loss + topo_scale * (W_CLDICE*l_cldice + W_SCNP*l_scnp + W_CONTAIN*l_contain)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item(); tb += 1

        sched.step()

        model.eval(); ap = []; at = []
        with torch.no_grad():
            for imgs, seg_masks, _ in val_dl:
                imgs = imgs.to(device)
                seg_logits, _ = model(imgs)
                ap.append(seg_logits.argmax(1).cpu())
                at.append(seg_masks)

        preds = torch.cat(ap); tgts = torch.cat(at)
        ious  = [(((preds==c)&(tgts==c)).sum().float()+1e-6) /
                 (((preds==c)|(tgts==c)).sum().float()+1e-6) for c in range(3)]
        miou  = np.mean([x.item() for x in ious])

        peak_mb = (torch.cuda.max_memory_allocated()/1024**2) if device.type=='cuda' else 0
        if (ep+1) % 5 == 0 or miou > best:
            print(f"  Ep {ep+1:3d}/{EPOCHS} | loss={tl/tb:.4f} | "
                  f"mIoU={miou:.4f} epi={ious[1]:.3f} lum={ious[2]:.3f} | "
                  f"{(time.time()-t0)/60:.1f}m | topo={topo_scale:.2f} | "
                  f"VRAM_peak={peak_mb:.0f}MB")
        if miou > best:
            best = miou; pat = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'encoder':   'efficientnet-b4',
                    'architecture': 'UNet_FullImage_ShapeConstrained_SCSE',
                    'img_size':  512,
                    'losses':    'Dice+CE+clDice+SCNP+Containment',
                    'attention': 'scse',
                    'stain_aug': 'macenko_sigma0.2' if stain_aug else 'none',
                    'gradient_checkpointing': 'encoder_full+decoders',
                    'topo_weights': {'cldice': W_CLDICE, 'scnp': W_SCNP, 'containment': W_CONTAIN},
                    'topo_warmup':  'linear_20_40',
                    'from_scratch': True,
                    'clean_dataset': True,
                    'mask_source': 'masks_v2/masks (regenerate, +107 con --no-is-clean)',
                    'version':   'v23'
                },
                'best_val_iou': best
            }, str(model_path))
            print(f"    * MEJOR: {best:.4f}")
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  Early stopping at epoch {ep+1} (best={best:.4f})")
            break

    print(f"\n{'='*60}\n  COMPLETE in {(time.time()-t0)/60:.1f} min — mIoU: {best:.4f}\n{'='*60}")
    try:
        from huggingface_hub import HfApi; api = HfApi()
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo='best_model_v23.pt',
            repo_id='LuGot16/seminiferous-tubule-segmentation',
            repo_type='model')
        print("  Upload to HF!")
    except Exception as e:
        print(f"  Upload: {e}")

if __name__ == '__main__':
    main()
