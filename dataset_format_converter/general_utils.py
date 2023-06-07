import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from base_models_classes.empathy_kind import EmpathyKindClassifier, EmpathyKindEnum
from base_models_classes.exist_empathy import ExistEmpathyClassifier


class DialogueFunctions:

    @classmethod
    def number_of_party(cls, df):
        pass


class EmpathyFunctions:
    EMPATHY_KIND_MODULE = EmpathyKindClassifier()
    EMPATHY_EXIST_MODULE = ExistEmpathyClassifier()
    EMPATHY_KIND_SEQUENCE = f".*(({EmpathyKindEnum.SEEKING.value}, )+({EmpathyKindEnum.NONE.value}, )*({EmpathyKindEnum.PROVIDING.value}(, )?)+)+.*"

    @classmethod
    def get_empathy_kind(cls,
                         data: pd.DataFrame,
                         utter_key_name='utterance',
                         result_key_name='empathy_kind') -> pd.DataFrame:
        """
        apply EMPATHY_KIND_MODULE
        :param result_key_name: name of col that gonna add as result of module
        :param data:
        :param utter_key_name: name of col that have utterance values
        :return: dataframe contains result col
        """
        utterances = data[[utter_key_name]].to_numpy()
        data[result_key_name] = cls.EMPATHY_KIND_MODULE(utterances)
        return data

    @classmethod
    def get_empathy_exist(cls,
                          data: pd.DataFrame,
                          utter_key_name='utterance',
                          conv_id_key_name='conv_id',
                          utter_id_key_name='utterance_idx',
                          result_key_name='is_empathy') -> pd.DataFrame:
        """
        apply EMPATHY_EXIST_MODULE
        :param data:
        :param utter_key_name: name of col that have utterance values
        :param conv_id_key_name: name of col that have conv_id values
        :param utter_id_key_name:
        :param result_key_name: name of col that gonna add as result of module
        :return: dataframe contains result col
        """
        conv_df = data.sort_values(by=[conv_id_key_name, utter_id_key_name])
        conv_df = conv_df.groupby(conv_id_key_name)[utter_key_name].apply(list).reset_index()
        conversations = conv_df[[utter_key_name]].to_numpy()
        conv_df[result_key_name] = cls.EMPATHY_EXIST_MODULE(conversations)
        return conv_df[[conv_id_key_name, result_key_name]].merge(data, on=conv_id_key_name, how='inner')

    @classmethod
    def check_empathy_kind_seq(cls,
                               data: pd.DataFrame,
                               utter_key_name='utterance',
                               utter_id_key_name='utterance_idx',
                               conv_id_key_name='conv_id',
                               empathy_kind_key_name='empathy_kind',
                               empathy_seq_key_name='empathy_kind_seq',
                               result_key_name='contain_empathy_seq'):
        """
        check a sequence of empathy kind
        :param empathy_seq_key_name: it is gonna be new col that this func add it
        :param result_key_name:
        :param empathy_kind_key_name:
        :param data: must be dataframe
        :param utter_key_name:
        :param utter_id_key_name:
        :param conv_id_key_name:
        :return:
        """
        if empathy_kind_key_name not in data.columns:
            data = cls.get_empathy_kind(data, utter_key_name=utter_key_name, result_key_name=empathy_kind_key_name)

        df_copy = data.sort_values(by=[conv_id_key_name, utter_id_key_name]).copy()
        df_copy[empathy_kind_key_name] = df_copy[empathy_kind_key_name].apply(str)
        conv_df = df_copy[[conv_id_key_name, empathy_kind_key_name]].\
            groupby([conv_id_key_name])[empathy_kind_key_name].apply(', '.join).reset_index().\
            rename(columns={empathy_kind_key_name: empathy_seq_key_name})
        conv_df[result_key_name] = conv_df[empathy_seq_key_name].str.match(cls.EMPATHY_KIND_SEQUENCE)
        return conv_df[[conv_id_key_name, empathy_seq_key_name, result_key_name]].merge(data,
                                                                                        on=conv_id_key_name,
                                                                                        how='inner')

    @classmethod
    def segment_empathy_dialogue(cls,
                                 data,
                                 utter_key_name='utterance',
                                 utter_id_key_name='utterance_idx',
                                 conv_id_key_name='conv_id',
                                 empathy_kind_key_name='empathy_kind'):
        """:param
        برای سگمنت دو تا چیز مهمه یکی اینکه باید بری نوع همدلی و یکی همگام کنی این دوستان عزیز رو با اشخاصی که طرف روبه رو عه
        یعنی مکالمه از طرف یکی شروع میشه که جویای همدلیه و با یه فرد مقابل اولیه تموم میشه که همدلی رو تهیه میکنه یا کلا با شخص رو به رو واکنش همدلی نشون میده
        اینو به این مدلی که همدلی وجود داره یا نه هم باید بدی که ببینی همدلی وجود داره یا نه چون ممکنه نسبت به فرد مقابل و داده هاش همدلی نشون نده
        بعد یه تگ جدیدی میسازی که ببینی برای آیدی مکالمه جدید و ترتیب انجام گفتگو
        """
        pass
