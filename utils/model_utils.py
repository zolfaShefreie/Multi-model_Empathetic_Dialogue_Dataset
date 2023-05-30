import torch
from abc import ABC, abstractmethod


class BaseDeployedModel(ABC):

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
        self.model_class = self._get_model_class()
        self.model = self.model_class.load_from_checkpoint(self._get_checkpoint_path(),
                                                           map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                                     else 'cpu'))

    def predict(self, data):
        """
        :param data:
        :return:
        """
        processed_data = self.data_pre_process_pipeline(data)
        model_output = self.model(processed_data)
        return self.result_after_process_pipeline(model_output)

    def __call__(self, data):
        return self.predict(data)
