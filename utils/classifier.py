import torch
import lightning.pytorch as pl

from base_models_classes.empathy_kind import EmpathyKindRobertaModel
from settings import PREFIX_CLASSIFIER_DIR


class Classifier:
    
    CLASSIFIER_INFO = {
        "empathy_kind": {
            'checkpoint_path': f"{PREFIX_CLASSIFIER_DIR}",
            'class': EmpathyKindRobertaModel
        },
        "exist_empathy": {
            'checkpoint_path': f"{PREFIX_CLASSIFIER_DIR}",
            'class': None
        }
    }

    def __init__(self, classifier_name: str):
        if classifier_name not in self.CLASSIFIER_INFO.keys():
            raise Exception("invalid classifier_name")

        self.trainer = pl.Trainer(accelerator='auto')

        # load model
        class_model = self.CLASSIFIER_INFO[classifier_name]['class']
        self.model = class_model.load_from_checkpoint(self.CLASSIFIER_INFO[classifier_name]['checkpoint_path'],
                                                      map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # TODO: write how to get data and send to transformers and make dataloader and predict
    def predict(self, dataloader):
        pass

