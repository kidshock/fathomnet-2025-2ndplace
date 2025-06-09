import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models


class FathomNetHierarchicalClassifier(pl.LightningModule):
    def __init__(self, num_classes, distance_matrix, lr, weight_decay, h_loss_weight, label_smoothing, mixup_alpha):
        super().__init__()
        self.save_hyperparameters(ignore=['distance_matrix'])
        self.register_buffer("distance_matrix", distance_matrix.float(), persistent=False)

        # ROI Stream Backbone (EfficientNetV2-M)
        roi_backbone = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        roi_feat_dim = roi_backbone.classifier[1].in_features
        roi_backbone.classifier = nn.Identity()
        self.roi_backbone = roi_backbone

        # Full Image Stream Backbone (EfficientNetV2-S)
        full_backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        full_feat_dim = full_backbone.classifier[1].in_features
        full_backbone.classifier = nn.Identity()
        self.full_image_backbone = full_backbone

        combined_feat_dim = roi_feat_dim + full_feat_dim

        # Gate heads for confidence estimation
        self.roi_gate_head = nn.Linear(roi_feat_dim, num_classes)
        self.full_gate_head = nn.Linear(full_feat_dim, num_classes)

        # Gate network for dynamic fusion
        self.gate_network = nn.Sequential(
            nn.Linear(2, combined_feat_dim),
            nn.Sigmoid()
        )

        # Final classifier head
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(combined_feat_dim),
            nn.Dropout(p=0.4),
            nn.Linear(combined_feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # Loss function
        self.ce = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)

    def mixup_data(self, x_tuple, y):
        if self.hparams.mixup_alpha > 0:
            lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
        else:
            lam = 1.0

        x_roi, x_full = x_tuple
        batch_size = x_roi.size(0)
        index = torch.randperm(batch_size, device=self.device)

        mixed_x_roi = lam * x_roi + (1 - lam) * x_roi[index]
        mixed_x_full = lam * x_full + (1 - lam) * x_full[index]
        y_a, y_b = y, y[index]

        return (mixed_x_roi, mixed_x_full), y_a, y_b, lam

    def forward(self, x_tuple):
        x_roi, x_full = x_tuple

        # Extract features
        roi_features = self.roi_backbone(x_roi)
        full_features = self.full_image_backbone(x_full)

        # Confidence estimation (entropy-based)
        roi_logits = self.roi_gate_head(roi_features)
        full_logits = self.full_gate_head(full_features)

        roi_probs = F.softmax(roi_logits, dim=1)
        full_probs = F.softmax(full_logits, dim=1)

        roi_entropy = -torch.sum(roi_probs * torch.log(roi_probs + 1e-8), dim=1, keepdim=True)
        full_entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-8), dim=1, keepdim=True)

        max_entropy = np.log(roi_probs.size(1))
        roi_conf = 1.0 - roi_entropy / max_entropy
        full_conf = 1.0 - full_entropy / max_entropy

        gate_input = torch.cat([roi_conf, full_conf], dim=1)
        gate = self.gate_network(gate_input)

        # Fuse and classify
        combined_features = torch.cat((roi_features, full_features), dim=1)
        gated_features = combined_features * gate
        logits = self.classifier_head(gated_features)

        return logits

    def training_step(self, batch, batch_idx):
        x_tuple, y = batch
        mixed_x, y_a, y_b, lam = self.mixup_data(x_tuple, y)
        logits = self(mixed_x)

        # MixUp loss
        ce_loss = lam * self.ce(logits, y_a) + (1 - lam) * self.ce(logits, y_b)

        # Hierarchical penalty
        probs = F.softmax(logits, dim=1)
        dist = self.distance_matrix.to(self.device)
        h_penalty = lam * (probs * dist[y_a]).sum(dim=1).mean() + \
                    (1 - lam) * (probs * dist[y_b]).sum(dim=1).mean()

        loss = ce_loss + self.hparams.h_loss_weight * h_penalty

        self.log_dict({
            "train_loss": loss,
            "train_ce": ce_loss,
            "train_h_penalty": h_penalty
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tuple, y = batch
        logits = self(x_tuple)
        val_ce = self.ce(logits, y)

        preds = torch.argmax(logits, dim=1)
        dist = self.distance_matrix.to(self.device)
        h_dist = dist[y, preds].mean()

        self.log_dict({
            "val_loss": val_ce,
            "val_hdist": h_dist
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches if self.trainer.max_steps == -1 else self.trainer.max_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
