import os
import time
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# Project imports
from . import config
from . import utils
from . import dataset
from . import model


class EpochMetricsPrinter(pl.Callback):
    """Custom callback to print metrics after each validation epoch."""
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics

        fold_info = f"Fold {trainer.fold_idx + 1}/{config.N_FOLDS} - "
        train_loss = metrics.get('train_loss_epoch', float('nan'))
        val_loss = metrics.get('val_loss', float('nan'))
        val_hdist = metrics.get('val_hdist', float('nan'))

        print(f"\n{fold_info}Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val H-Dist={val_hdist:.4f}")


def run_training():
    """Run K-Fold cross-validation training pipeline."""
    utils.seed_everything(config.SEED)

    if config.ACCELERATOR == "gpu":
        torch.set_float32_matmul_precision('high')

    print("--- FathomNet 2025 Training Pipeline ---")
    print(f"Using strategy: {config.STRATEGY} with {config.DEVICES} device(s)")

    # 1. Load dataset and encode labels
    print("Loading dataset...")
    full_train_df = pd.read_csv(config.TRAIN_ANNOTATIONS_CSV)

    # Optional: full_train_df['full_image_path'] = full_train_df['path'].apply(...)
    label_encoder = LabelEncoder()
    full_train_df['label_encoded'] = label_encoder.fit_transform(full_train_df['label'])
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} classes.")

    # 2. Load or compute hierarchical distance matrix
    print("Preparing distance matrix...")
    distance_matrix = utils.get_distance_matrix(
        label_encoder,
        taxonomy_path=config.TAXONOMY_TREE_PATH,
        save_path=config.DISTANCE_MATRIX_PATH,
        max_dist=config.MAX_DISTANCE
    )

    # 3. Cross-validation setup
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(full_train_df, full_train_df['label_encoded'])):
        print(f"\n{'='*20} FOLD {fold_idx + 1}/{config.N_FOLDS} {'='*20}")

        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]

        # 4. Prepare Datasets and Loaders
        train_ds = dataset.FathomNetDataset(
            train_df,
            transform_roi=dataset.train_transforms_roi,
            transform_full=dataset.full_image_transforms,
            label_encoder=label_encoder
        )
        val_ds = dataset.FathomNetDataset(
            val_df,
            transform_roi=dataset.val_test_transforms_roi,
            transform_full=dataset.full_image_transforms,
            label_encoder=label_encoder
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=utils.seed_worker
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=utils.seed_worker
        )

        # 5. Define Callbacks
        checkpoint_dir = os.path.join(config.CHECKPOINT_BASE_DIR, f"fold_{fold_idx}")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_hdist",
            mode="min",
            save_top_k=1,
            filename="best-hdist-{epoch:02d}-{val_hdist:.4f}",
            dirpath=checkpoint_dir
        )
        early_stop_cb = EarlyStopping(
            monitor="val_hdist",
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.MIN_DELTA,
            mode="min"
        )
        logger = CSVLogger(config.LOGS_DIR, name=f"{config.MODEL_NAME}-fold{fold_idx}")

        # 6. Trainer setup
        trainer = pl.Trainer(
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            strategy=config.STRATEGY,
            max_epochs=config.MAX_EPOCHS,
            precision=config.PRECISION,
            logger=logger,
            callbacks=[checkpoint_cb, early_stop_cb, EpochMetricsPrinter(), TQDMProgressBar(refresh_rate=10)],
            log_every_n_steps=10
        )
        trainer.fold_idx = fold_idx

        # 7. Initialize model
        model_instance = model.FathomNetHierarchicalClassifier(
            num_classes=num_classes,
            distance_matrix=distance_matrix,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            h_loss_weight=config.H_LOSS_WEIGHT,
            label_smoothing=config.LABEL_SMOOTHING,
            mixup_alpha=config.MIXUP_ALPHA
        )

        print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        # 8. Train the model
        trainer.fit(model_instance, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_score = checkpoint_cb.best_model_score.item()
        oof_scores.append(best_score)
        print(f"Fold {fold_idx + 1} Best Val H-Dist: {best_score:.4f}")

    # Final summary
    print(f"\n{'='*20} TRAINING COMPLETE {'='*20}")
    print(f"Average OOF Validation H-Dist across {config.N_FOLDS} folds: {np.mean(oof_scores):.4f}")


if __name__ == "__main__":
    run_training()
# This script is intended to be run as a standalone module.