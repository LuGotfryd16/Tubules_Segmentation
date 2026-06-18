"""
=================================================================
INFERENCE v1.0.0 — Full-image + SCSE attention + TTA
=================================================================
"""

import os, cv2, torch, numpy as np, json, time, csv
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from pathlib import Path

# ============================================================
# MODEL v1.0.0
# ============================================================
class TubuleSegModel(nn.Module):
    def __init__(self, attention_type='scse'):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            'timm-efficientnet-b4', in_channels=3, depth=5, weights=None
        )
        ec = self.encoder.out_channels
        self.seg_decoder = UnetDecoder(
            encoder_channels=ec, decoder_channels=(256,128,64,32,16),
            n_blocks=5, use_norm='batchnorm', attention_type=attention_type
        )
        self.seg_head = nn.Conv2d(16, 3, kernel_size=1)
        self.border_decoder = UnetDecoder(
            encoder_channels=ec, decoder_channels=(256,128,64,32,16),
            n_blocks=5, use_norm='batchnorm', attention_type=attention_type
        )
        self.border_head = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return (
            self.seg_head(self.seg_decoder(features)),
            self.border_head(self.border_decoder(features))
        )


# ============================================================
# TTA — Test-Time Augmentation
# ============================================================
def predict_with_tta(model, img_t, device):
    """
    8 pasadas: 4 rotaciones (0, 90, 180, 270) x 2 flips (original, horizontal).
    Promedia probabilidades softmax en el espacio original.
    img_t: [1, 3, H, W] tensor normalizado en device.
    Retorna: [3, H, W] numpy array de probabilidades promediadas.
    """
    preds = []
    with torch.no_grad():
        for flip in [False, True]:
            x = img_t.flip(-1) if flip else img_t
            for k in range(4):  # rotaciones 0, 90, 180, 270
                x_rot = torch.rot90(x, k, dims=[-2, -1])
                seg_logits, _ = model(x_rot)
                prob = F.softmax(seg_logits, dim=1)  # [1, 3, H, W]
                # Revertir la rotacion
                prob = torch.rot90(prob, -k, dims=[-2, -1])
                # Revertir el flip
                if flip:
                    prob = prob.flip(-1)
                preds.append(prob.cpu().numpy()[0])  # [3, H, W]

    return np.mean(preds, axis=0)  # [3, H, W]


# ============================================================
# POST-PROCESSING (identico a inference_v22.py)
# ============================================================
def postprocess(pred, h, w):
    epi_mask = (pred == 1).astype(np.uint8)
    n, labels = cv2.connectedComponents(epi_mask)
    if n > 1:
        areas = [(lid, (labels==lid).sum()) for lid in range(1, n)]
        areas.sort(key=lambda x: x[1], reverse=True)
        epi_clean = (labels == areas[0][0]).astype(np.uint8)
    else:
        epi_clean = epi_mask

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    epi_closed = cv2.morphologyEx(epi_clean, cv2.MORPH_CLOSE, k_close)

    flood = epi_closed.copy()
    fm = np.zeros((h+2, w+2), dtype=np.uint8)
    for y in range(0, h, 3):
        if flood[y, 0] == 0:   cv2.floodFill(flood, fm, (0,   y), 2)
        if flood[y, w-1] == 0: cv2.floodFill(flood, fm, (w-1, y), 2)
    for x in range(0, w, 3):
        if flood[0, x] == 0:   cv2.floodFill(flood, fm, (x,   0), 2)
        if flood[h-1, x] == 0: cv2.floodFill(flood, fm, (x, h-1), 2)
    holes = (flood == 0).astype(np.uint8)
    epi_filled = epi_closed | holes

    final = np.zeros((h, w), dtype=np.uint8)
    final[epi_filled > 0] = 1

    lum_mask = (pred == 2).astype(np.uint8)
    lum_inside = lum_mask & epi_filled
    n, labels = cv2.connectedComponents(lum_inside)
    if n > 1:
        lum_clean = np.zeros_like(lum_inside)
        for lid in range(1, n):
            comp = (labels == lid).astype(np.uint8)
            if comp.sum() < 300: continue
            dilated = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            border_px = dilated & (~epi_filled.astype(bool)).astype(np.uint8)
            if border_px.sum() < comp.sum() * 0.05:
                lum_clean[labels == lid] = 1
        final[lum_clean > 0] = 2

    return final


# ============================================================
# MAIN
# ============================================================
def main():
    base_dir = Path(r"D:\Lu\AI\Tubules"); os.chdir(str(base_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_file = 'best_model_segmentation_v1.0.0.pt'
    attention  = 'scse'

    model = TubuleSegModel(attention_type=attention)
    ckpt  = torch.load(str(base_dir / model_file), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device); model.eval()
    
    from huggingface_hub import snapshot_download
    data_repo = base_dir / 'data_repo'
    if not data_repo.exists():
        snapshot_download('LuGot16/tubules', repo_type='dataset', local_dir=str(data_repo))

    test_dir = data_repo / 'area_test'
    out_dir  = base_dir / f'inference_results_{model_file.replace("best_model_","").replace(".pt","")}'
    out_dir.mkdir(exist_ok=True)

    SCALE   = 0.32  # um/pixel
    results = []
    files   = sorted(test_dir.glob('*.tif'))
    t0 = time.time()

    for i, fp in enumerate(files):
        img = cv2.imread(str(fp))
        if img is None: continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_norm = (img_resized.astype(np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img_t = torch.from_numpy(img_norm.transpose(2,0,1)).float().unsqueeze(0).to(device)

        # TTA
        seg_probs = predict_with_tta(model, img_t, device)  # [3, 512, 512]

        pred_512 = seg_probs.argmax(0).astype(np.uint8)
        pred     = cv2.resize(pred_512, (w, h), interpolation=cv2.INTER_NEAREST)
        pred     = postprocess(pred, h, w)

        total   = h * w
        epi_pct = (pred==1).sum() / total * 100
        lum_pct = (pred==2).sum() / total * 100
        tub_pct = epi_pct + lum_pct
        epi_um2 = (pred==1).sum() * SCALE * SCALE
        lum_um2 = (pred==2).sum() * SCALE * SCALE
        tub_um2 = epi_um2 + lum_um2
        status  = "OK" if tub_pct >= 30 and lum_pct > 1 else "CHECK"

        results.append({
            'image':          fp.name,
            'tubule_pct':     round(tub_pct, 1),
            'epithelium_pct': round(epi_pct, 1),
            'lumen_pct':      round(lum_pct, 1),
            'tubule_um2':     round(tub_um2, 1),
            'epithelium_um2': round(epi_um2, 1),
            'lumen_um2':      round(lum_um2, 1),
            'lumen_epi_ratio': round(lum_um2 / (epi_um2 + 1e-6), 4),
            'status':         status
        })

        # Overlay
        overlay = img.copy()
        overlay[pred==1] = (overlay[pred==1]*0.5 + np.array([0,180,0])*0.5).astype(np.uint8)
        overlay[pred==2] = (overlay[pred==2]*0.5 + np.array([255,100,0])*0.5).astype(np.uint8)

        tubule_mask = (pred >= 1).astype(np.uint8)
        contours_outer, _ = cv2.findContours(tubule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_outer, -1, (0, 255, 255), 2)

        lum_mask_vis = (pred == 2).astype(np.uint8)
        contours_lumen, _ = cv2.findContours(lum_mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_lumen, -1, (0, 0, 255), 2)

        cv2.imwrite(str(out_dir / f"{fp.stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{fp.stem}_mask.png"), pred * 127)

        if (i+1) % 10 == 0 or status == "CHECK":
            print(f"  [{i+1}/{len(files)}] {fp.name}: "
                  f"tub={tub_pct:.1f}% epi={epi_pct:.1f}% lum={lum_pct:.1f}% [{status}]")

    ok = sum(1 for r in results if r['status'] == 'OK')
    tub_vals = [r['tubule_pct']     for r in results]
    epi_vals = [r['epithelium_pct'] for r in results]
    lum_vals = [r['lumen_pct']      for r in results]

    print(f"\n{'='*60}")
    print(f"  RESULTS {model_file} + TTA — {ok}/{len(results)} OK")
    print(f"  Tubule:     {np.mean(tub_vals):.1f} +/- {np.std(tub_vals):.1f}%")
    print(f"  Epithelium:   {np.mean(epi_vals):.1f} +/- {np.std(epi_vals):.1f}%")
    print(f"  Lumen:      {np.mean(lum_vals):.1f} +/- {np.std(lum_vals):.1f}%")
    print(f"  Time:     {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")

    with open(out_dir / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'model': model_file, 'tta': True, 'n_tta': 8,
            'mIoU': float(ckpt['best_val_iou']),
            'n_ok': ok, 'n_total': len(results),
            'mean_tubule_pct': round(np.mean(tub_vals), 1),
            'mean_lumen_pct':  round(np.mean(lum_vals), 1),
            'results': results
        }, f, indent=2)
   


if __name__ == '__main__':
    main()
