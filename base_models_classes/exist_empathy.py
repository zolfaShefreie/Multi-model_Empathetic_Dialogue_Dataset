from transformers import RobertaModel, RobertaTokenizer
import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score

from utils import model_utils
from utils import util_transforms
from settings import EMPATHY_EXIST_MODEL_FILE_PATH

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

    def predict_step(self, batch:dict, batch_idx, dataloader_idx=0):
        return self(**batch)


class ExistEmpathyClassifier(model_utils.BaseDeployedModel):

    def _get_checkpoint_path(self) -> str:
        """
        :return: path of model checkpoint
        """
        return EMPATHY_EXIST_MODEL_FILE_PATH

    def _get_model_class(self):
        """
        :return: model class
        """
        return EmpathyDetectionRobertaModel

    def _get_data_pre_process_list(self) -> list:
        """
        :return: a list of process to apply to input data
        """
        return [
            util_transforms.TextCleaner(have_label=False),
            util_transforms.ConversationFormatter(have_label=False),
            util_transforms.Tokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                      new_special_tokens={
                                          'additional_special_tokens': [
                                              util_transforms.ConversationFormatter.SPECIAL_TOKEN_START_UTTERANCE,
                                              util_transforms.ConversationFormatter.SPECIAL_TOKEN_END_UTTERANCE],
                                          'pad_token': '[PAD]'},
                                      max_len=512,
                                      have_label=False),
            util_transforms.ToTensor(),
        ]

    def _get_result_after_process_list(self) -> list:
        """
        :return: a list of process to apply to model output
        """
        return [
            torch.nn.functional.sigmoid,
            util_transforms.IntegerConverterWithThreshold(threshold=0.5)
        ]

    def get_arg_index_model_input(self):
        """
        :return: kwargs format of model()
        """
        return {
            'ids': 0,
            'mask': 1,
            'token_type_ids': 2
        }

