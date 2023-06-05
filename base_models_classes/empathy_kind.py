from transformers import RobertaModel, RobertaTokenizer
import torch
import lightning.pytorch as pl
import torchmetrics
import enum

from utils import model_utils
from utils import util_transforms
from settings import PREFIX_CLASSIFIER_DIR


class EmpathyKindEnum(enum.Enum):
    NONE = 0
    SEEKING = 1
    PROVIDING = 2


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

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0):
        return self(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class EmpathyKindClassifier(model_utils.BaseDeployedModel):

    def __init__(self, have_batch_d=True):
        self.have_batch_d = have_batch_d
        super().__init__()

    def _get_checkpoint_path(self) -> str:
        """
        :return: path of model checkpoint
        """
        return f"{PREFIX_CLASSIFIER_DIR}/empathy_kind_classifier.ckpt"

    def _get_model_class(self):
        """
        :return: model class
        """
        return EmpathyKindRobertaModel

    def _get_data_pre_process_pipeline(self) -> util_transforms.Pipeline:
        """
        :return: a pipeline to preprocess input data
        """
        process = [
            util_transforms.TextCleaner(have_label=False),
            util_transforms.Tokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                      have_label=False),
            util_transforms.ToTensor(),
        ]

        if not self.have_batch_d:
            process.append(util_transforms.AddBatchDimension())

        process.append(util_transforms.ConvertInputToDict(dict_meta_data={
                'ids': 0,
                'mask': 1,
                'token_type_ids': 2
            }))

        return util_transforms.Pipeline(process)

    def _get_result_after_process_pipeline(self) -> util_transforms.Pipeline:
        """
        :return: a pipeline to apply
        """
        process = [
            torch.nn.functional.sigmoid,
        ]
        if not self.have_batch_d:
            process.append(util_transforms.DelBatchDimension())
        process.append(util_transforms.IntegerConverterWithIndex(dim=1 if self.have_batch_d else 0))

        return util_transforms.Pipeline(process)
