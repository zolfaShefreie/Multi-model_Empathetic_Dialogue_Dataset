import pandas as pd

from base_models_classes.empathy_kind import EmpathyKindClassifier
from base_models_classes.exist_empathy import ExistEmpathyClassifier


class EmpathyFunctions:
    EMPATHY_KIND_MODULE = EmpathyKindClassifier(have_batch_d=True)
    EMPATHY_EXIST_MODULE = ExistEmpathyClassifier(have_batch_d=True)
    MAX_BATCH_SIZE = 50

    @classmethod
    def _get_result(cls, input_data, module):
        #TODO: complete
        results = list()
        return input_data

    @classmethod
    def get_empathy_kind(cls,
                         data: pd.DataFrame,
                         utter_key_name='utterance',
                         result_key_name='empathy_kind') -> pd.DataFrame:
        """
        :param result_key_name:
        :param data:
        :param utter_key_name:
        :return:
        """
        utterances = data[[utter_key_name]].to_numpy()
        data[result_key_name] = cls._get_result(input_data=utterances, module=cls.EMPATHY_KIND_MODULE)
        return data

    @classmethod
    def get_empathy_exist(cls,
                          data: pd.DataFrame,
                          utter_key_name='utterance',
                          conv_id_key_name='conv_id',
                          result_key_name='is_empathy') -> pd.DataFrame:
        """

        :param data:
        :param utter_key_name:
        :param conv_id_key_name:
        :param result_key_name:
        :return:
        """
        conv_df = data.groupby(conv_id_key_name)[utter_key_name].apply(list).reset_index()
        conversations = conv_df[[utter_key_name]].to_numpy()
        conv_df[result_key_name] = cls._get_result(input_data=conversations, module=cls.EMPATHY_EXIST_MODULE)
        return conv_df[[conv_id_key_name, result_key_name]].merge(data, on=conv_id_key_name, how='inner')

    @classmethod
    def detect(cls,
               data: pd.DataFrame,
               utter_key_name='utterance',
               utter_id_key_name='utterance_idx',
               conv_id_key_name='conv_id'):
        """

        :param data: must be dataframe
        :param utter_key_name:
        :param utter_id_key_name:
        :param conv_id_key_name:
        :return:
        """
        pass

    @classmethod
    def segment_empathy_dialogue(cls,
                                 data,
                                 utter_key_name='utterance',
                                 utter_id_key_name='utterance_idx',
                                 conv_id_key_name='conv_id',
                                 empathy_kind_key_name='empathy_kind'):
        pass
