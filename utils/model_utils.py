import torch
from abc import ABC, abstractmethod
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
import pandas as pd
import numpy as np

from utils.util_transforms import Pipeline, ConvertInputToDict, AddBatchDimension, DelBatchDimension


class BaseDataset(Dataset):
    def __init__(self, data, pipeline_transforms=None):
        self.x = data
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
        """
        :return: path of model checkpoint
        """
        raise NotImplementedError

    @abstractmethod
    def _get_model_class(self):
        """
        :return: model class
        """
        raise NotImplementedError

    @abstractmethod
    def _get_data_pre_process_list(self) -> list:
        """"
        :return: a list of process to apply to input data
        """
        raise NotImplementedError

    @abstractmethod
    def _get_result_after_process_list(self) -> list:
        """
        :return: a list of process to apply to model output
        """
        raise NotImplementedError

    @abstractmethod
    def get_arg_index_model_input(self):
        """
        :return: kwargs format of model()
        """
        raise NotImplementedError

    def __init__(self):

        self.data_pre_process_list = self._get_data_pre_process_list()
        self.result_after_process_list = self._get_result_after_process_list()
        self.input_dict_format = self.get_arg_index_model_input()

        # load model
        self.model_class = self._get_model_class()
        self.model = self.model_class.load_from_checkpoint(self._get_checkpoint_path(),
                                                           map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                                     else 'cpu'))
        self.model.eval()
        self.trainer = Trainer(self.model)

    def _make_preprocess_pipeline(self, is_single=False):
        """
        :param is_single: a bool that shows data is a single sample or multisample
        :return: pipeline
        """
        if is_single:
            return Pipeline(self.data_pre_process_list +
                            [AddBatchDimension(), ConvertInputToDict(dict_meta_data=self.input_dict_format)])

        return Pipeline(self.data_pre_process_list + [ConvertInputToDict(dict_meta_data=self.input_dict_format)])

    def _make_result_after_process_pipeline(self, is_single=False):
        """
            :param is_single: a bool that shows data is a single sample or multisample
            :return: pipeline
        """
        if is_single:
            return Pipeline(self.result_after_process_list + [DelBatchDimension(), ])

        return Pipeline(self.result_after_process_list)

    def _predict_dataset(self, data):
        """
        :param data:
        :return:
        """
        dataset = BaseDataset(data, pipeline_transforms=self._make_preprocess_pipeline(is_single=False))
        y_hat = self.trainer.predict(dataset).predictions
        return self._make_result_after_process_pipeline(is_single=False)(torch.from_numpy(y_hat))

    def _predict_single_data(self, data):
        """
        :param data:
        :return:
        """
        processed_data = self._make_preprocess_pipeline(is_single=True)(data)
        model_output = self.model(**processed_data) if isinstance(processed_data, dict) else self.model(processed_data)
        return self._make_result_after_process_pipeline(is_single=True)(model_output)

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
