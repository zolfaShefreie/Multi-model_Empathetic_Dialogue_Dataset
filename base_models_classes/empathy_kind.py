from transformers import RobertaModel
import torch
import lightning.pytorch as pl
import torchmetrics


class EmpathyKindRobertaModel(pl.LightningModule):
    num_classes = 3
    LOSS = torch.nn.BCEWithLogitsLoss()

    def __init__(self):
        super().__init__()
        self.transformer_model = RobertaModel.from_pretrained("roberta-base")
        self.drop = torch.nn.Dropout(0.4)
        self.out = torch.nn.Linear(768, self.num_classes)

    def forward(self, ids, mask, token_type_ids):
        x = self.transformer_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[0].mean(dim=1)
        x = self.drop(x)
        output = self.out(x)

        return output

    def training_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_loss": loss, "train_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        val_loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        self.log_dict({"val_loss": val_loss, "val_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        test_loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        f1_score = torchmetrics.functional.classification.multilabel_f1_score(pred, y.float(), num_labels=3)
        self.log_dict({"test_loss": test_loss, "test_accuracy": acc, "test_f1": f1_score},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids, mask, token_type_ids = batch
        result = self(ids, mask, token_type_ids)
        return torch.nn.functional.sigmoid(result)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

