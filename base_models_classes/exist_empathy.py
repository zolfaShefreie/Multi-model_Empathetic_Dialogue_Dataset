from transformers import RobertaModel, RobertaTokenizer
import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score

from utils import model_utils
from utils import dataset_transforms
from settings import PREFIX_CLASSIFIER_DIR

from transformers import RobertaModel


class EmpathyDetectionRobertaModel(pl.LightningModule):
    LOSS = torch.nn.BCEWithLogitsLoss()
    accuracy = Accuracy(task="binary").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    f1 = F1Score(task="binary").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def __init__(self, embedding_tokens_len=50268):
        super().__init__()
        self.transformer_model = RobertaModel.from_pretrained("roberta-base")
        if embedding_tokens_len:
            # when transformer_model.wte.weight.shape[0] != len(tokenizer)
            self.transformer_model.resize_token_embeddings(embedding_tokens_len)
        self.drop = torch.nn.Dropout(0.4)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        x = self.transformer_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[0].mean(dim=1)
        x = self.drop(x)
        output = self.out(x)

        return output

    def training_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        loss = self.LOSS(pred, y.float())
        acc = self.accuracy(pred, y.float())
        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_loss": loss, "train_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        val_loss = self.LOSS(pred, y.float())
        acc = self.accuracy(pred, y.float())
        self.log_dict({"val_loss": val_loss, "val_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        test_loss = self.LOSS(pred, y.float())
        acc = self.accuracy(pred, y.float())
        f1_score = self.f1(pred, y.float())
        self.log_dict({"test_loss": test_loss, "test_accuracy": acc, "test_f1": f1_score},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids, mask, token_type_ids = batch
        result = self(ids, mask, token_type_ids)
        return torch.nn.functional.sigmoid(result)

