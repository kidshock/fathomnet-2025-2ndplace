# 📄 `src/inference.py`

```python
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from sklearn.preprocessing import LabelEncoder

from . import config
from . import utils
from . import dataset
from . import model

ImageFile.LOAD_TRUNCATED_IMAGES = True

SLIGHT_ROTATION_ANGLE = 10

def get_model_weights(checkpoint_files: list) -> torch.Tensor:
    models_with_scores = []
    for ckpt_path in checkpoint_files:
        match = re.search(r"val_hdist=([0-9]+\.?[0-9]*)", os.path.basename(ckpt_path))
        if match:
            score = float(match.group(1))
        else:
            score = float(config.MAX_DISTANCE)
        
        models_with_scores.append({'path': ckpt_path, 'score': score})

    if not models_with_scores:
        raise ValueError("Could not parse scores from any checkpoint files.")

    raw_scores = np.array([m['score'] for m in models_with_scores])
    weights_np = 1.0 / (raw_scores + 1e-6)
    model_weights = torch.tensor(weights_np / np.sum(weights_np), dtype=torch.float32)
        
    return model_weights, [m['path'] for m in models_with_scores]

def run_inference(args):
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')

    train_df = pd.read_csv(config.TRAIN_ANNOTATIONS_CSV)
    global_label_encoder = LabelEncoder().fit(train_df['label'].unique())
    num_classes = len(global_label_encoder.classes_)

    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoints_dir, "**", "*.ckpt"), recursive=True))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt model files found in '{args.checkpoints_dir}'.")

    model_weights, ordered_model_paths = get_model_weights(checkpoint_files)
    model_weights = model_weights.to(device)

    test_df = pd.read_csv(config.TEST_ANNOTATIONS_CSV)

    # NOTE: You need to create the 'full_image_path' column here, same as in training.
    # Example preprocessing step (to be run once):
    # test_df['full_image_path'] = test_df['path'].apply(lambda p: ...)

    test_ds = dataset.FathomNetDataset(
        test_df,
        dataset.val_test_transforms_roi,
        dataset.full_image_transforms,
        global_label_encoder,
        is_test=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=utils.seed_worker
    )

    all_model_probs = []
    all_original_ids = []

    for model_idx, model_path in enumerate(ordered_model_paths):
        inference_model = model.FathomNetHierarchicalClassifier.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location="cpu"
        ).to(device).eval()

        model_accumulated_logits = []
        with torch.no_grad():
            for (roi_images, full_images), batch_ids in tqdm(test_loader, desc=f"Predict (M {model_idx+1})"):
                if model_idx == 0:
                    all_original_ids.extend(batch_ids.tolist())

                roi_images, full_images = roi_images.to(device), full_images.to(device)
                logits1 = inference_model((roi_images, full_images))
                roi_rot = TF.rotate(roi_images, angle=SLIGHT_ROTATION_ANGLE)
                full_rot = TF.rotate(full_images, angle=SLIGHT_ROTATION_ANGLE)
                logits2 = inference_model((roi_rot, full_rot))
                batch_avg_tta_logits = (logits1 + logits2) / 2.0
                model_accumulated_logits.append(batch_avg_tta_logits.cpu())

        model_probs = F.softmax(torch.cat(model_accumulated_logits), dim=1)
        all_model_probs.append(model_probs)

        del inference_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    stacked_probs = torch.stack(all_model_probs)
    ensembled_probs = torch.sum(stacked_probs * model_weights.cpu().view(-1, 1, 1), dim=0)
    final_preds_indices = torch.argmax(ensembled_probs, dim=1).numpy()
    decoded_concepts = global_label_encoder.inverse_transform(final_preds_indices)

    submission_df = pd.DataFrame({
        'annotation_id': all_original_ids,
        'concept_name': decoded_concepts
    })

    submission_df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FathomNet 2025 Inference Script")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=config.CHECKPOINT_BASE_DIR,
        help="Directory containing the trained K-Fold model checkpoints."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="submission.csv",
        help="Path to save the final submission CSV file."
    )
    args = parser.parse_args()

    run_inference(args)
