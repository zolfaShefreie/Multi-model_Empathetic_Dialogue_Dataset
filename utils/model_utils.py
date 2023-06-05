import torch
from abc import ABC, abstractmethod
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, data: pd.DataFrame, pipeline_transforms=None):
        self.x = data.to_numpy()
        self.n_sample = len(data)
        self.pipeline_transforms = pipeline_transforms

    def __getitem__(self, index):
        sample = self.x[index]
        if self.pipeline_transforms:
            sample = self.pipeline_transforms(sample)
        return sample

    def __len__(self):
        return self.n_sample


class BaseDeployedModel(ABC):
    MAX_BATCH_SIZE = 50

    @abstractmethod
    def _get_checkpoint_path(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_model_class(self):
        raise NotImplementedError

    @abstractmethod
    def _get_data_pre_process_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def _get_result_after_process_pipeline(self):
        raise NotImplementedError

    def __init__(self):

        self.data_pre_process_pipeline = self._get_data_pre_process_pipeline()
        self.result_after_process_pipeline = self._get_result_after_process_pipeline()

        # load model
        self.trainer = pl.Trainer(accelerator='auto')
        self.model_class = self._get_model_class()
        self.model = self.model_class.load_from_checkpoint(self._get_checkpoint_path(),
                                                           map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                                     else 'cpu'))
        self.model.eval()

    def _predict_dataset(self, data):
        """
        :param data:
        :return:
        """
        dataset = BaseDataset(data, pipeline_transforms=self.data_pre_process_pipeline)
        dataloader = DataLoader(dataset, batch_size=self.MAX_BATCH_SIZE)
        y_hat = self.trainer.predict(self.model, dataloader)
        return self.result_after_process_pipeline(torch.stack(y_hat, dim=0))

    def _predict_single_data(self, data):
        """
        :param data:
        :return:
        """
        processed_data = self.data_pre_process_pipeline(data)
        model_output = self.model(**processed_data) if isinstance(processed_data, dict) else self.model(processed_data)
        return self.result_after_process_pipeline(model_output)

    def predict(self, data, is_multi_data=True):
        """
        :param is_multi_data:
        :param data:
        :return:
        """
        if is_multi_data:
            return self._predict_dataset(data)
        # if isinstance(data, pd.DataFrame):
        #     return self._predict_dataset(data)

        return self._predict_single_data(data)

    def __call__(self, data, is_multi_data=True):
        return self.predict(data, is_multi_data=True)
