import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import re

from base_models_classes.empathy_kind import EmpathyKindClassifier, EmpathyKindEnum
from base_models_classes.exist_empathy import ExistEmpathyClassifier


class DialogueFunctions:

    @classmethod
    def number_of_party(cls,
                        data: pd.DataFrame,
                        conv_id_key_name='conv_id',
                        speaker_idx_key_name='speaker_idx',
                        result_key_name='num_party'):
        """
        calculate number of parties on conversation
        :param data:
        :param conv_id_key_name:
        :param speaker_idx_key_name:
        :param result_key_name:
        :return:
        """
        conv_df = data[[conv_id_key_name, speaker_idx_key_name]].groupby(conv_id_key_name)[speaker_idx_key_name].\
            apply(set).apply(len).reset_index().rename(columns={speaker_idx_key_name: result_key_name})
        return conv_df[[conv_id_key_name, result_key_name]].merge(data, on=conv_id_key_name, how='inner')

    @classmethod
    def filter_two_party_dialogues(cls,
                                   data: pd.DataFrame,
                                   conv_id_key_name='conv_id',
                                   speaker_idx_key_name='speaker_idx',
                                   num_parties_key_name='num_party'):
        """
        get conversations with two parties
        :param data:
        :param conv_id_key_name:
        :param speaker_idx_key_name:
        :param num_parties_key_name:
        :return: conversations with two parties as data dataframe
        """
        if num_parties_key_name not in data.columns:
            data = cls.number_of_party(data,
                                       conv_id_key_name=conv_id_key_name,
                                       speaker_idx_key_name=speaker_idx_key_name,
                                       result_key_name=num_parties_key_name)

        return data[data[num_parties_key_name] == 2]


class EmpathyFunctions:
    """
    this class is considered the dialogues are two-party
    """
    EMPATHY_KIND_MODULE = EmpathyKindClassifier()
    EMPATHY_EXIST_MODULE = ExistEmpathyClassifier()
    EMPATHY_KIND_SEQUENCE = f".*(({EmpathyKindEnum.SEEKING.value}, )(({EmpathyKindEnum.NONE.value}, )\1)*({EmpathyKindEnum.PROVIDING.value}(, )?))+.*"
    EMPATHY_KIND_SEGMENT_CONDITION = re.compile(f"(({EmpathyKindEnum.SEEKING.value})({EmpathyKindEnum.PROVIDING.value}))+")

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
                               result_key_name='contain_empathy_seq') -> pd.DataFrame:
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
        conv_df[result_key_name] = conv_df[empathy_seq_key_name].str.match(cls.EMPATHY_KIND_SEQUENCE).apply(int)
        return conv_df[[conv_id_key_name, empathy_seq_key_name, result_key_name]].merge(data,
                                                                                        on=conv_id_key_name,
                                                                                        how='inner')

    @classmethod
    def add_all_empathy_cols(cls,
                             data: pd.DataFrame,
                             utter_key_name='utterance',
                             utter_id_key_name='utterance_idx',
                             conv_id_key_name='conv_id',
                             empathy_kind_key_name='empathy_kind',
                             is_empathy_key_name='is_empathy'):
        """
        :param data:
        :param utter_key_name:
        :param utter_id_key_name:
        :param conv_id_key_name:
        :param empathy_kind_key_name:
        :param is_empathy_key_name:
        :return:
        """

        data = cls.get_empathy_kind(data=data,
                                    utter_key_name=utter_key_name,
                                    result_key_name=empathy_kind_key_name)
        return cls.get_empathy_exist(data=data,
                                     utter_key_name=utter_key_name,
                                     conv_id_key_name=conv_id_key_name,
                                     utter_id_key_name=utter_id_key_name,
                                     result_key_name=is_empathy_key_name)

    @classmethod
    def filter_empathetic_conversations(cls,
                                        data: pd.DataFrame,
                                        based_on='both',
                                        contain_empathy_key_name='contain_empathy_seq',
                                        is_empathy_key_name='is_empathy'):
        """

        :param data:
        :param based_on: can be 'both', 'contain_empathy', 'is_empathy'
        :param contain_empathy_key_name:
        :param is_empathy_key_name:
        :return:
        """
        if based_on == 'both':
            return data[((data[contain_empathy_key_name] == 1) & (data[is_empathy_key_name] == 1))]
        elif based_on == 'contain_empathy':
            return data[data[contain_empathy_key_name] == 1]
        elif based_on == 'is_empathy':
            return data[data[is_empathy_key_name] == 1]

        else:
            raise Exception('invalid based_on ')

    @classmethod
    def get_empathetic_user(cls,
                            data: pd.DataFrame,
                            speaker_idx_key_name='speaker_idx',
                            conv_id_key_name='conv_id',
                            utter_idx_key_name='utterance_idx',
                            empathy_kind_key_name='empathy_kind',
                            result_key_name='empathetic_speaker'):
        """
        get the user has the most empathy providing in one dialogue
        (this function is written for detect which user can be empathetic bot)
        :param result_key_name:
        :param data:
        :param speaker_idx_key_name:
        :param conv_id_key_name:
        :param utter_idx_key_name:
        :param empathy_kind_key_name:
        :return:
        """
        conv_df = data[data[empathy_kind_key_name] == EmpathyKindEnum.PROVIDING.value].\
            groupby([conv_id_key_name, speaker_idx_key_name, empathy_kind_key_name]).count().reset_index()
        conv_df = conv_df.loc[conv_df.reset_index().groupby([conv_id_key_name])[utter_idx_key_name].idxmax()].\
            rename(columns={speaker_idx_key_name: result_key_name})
        return conv_df[[conv_id_key_name, result_key_name]].merge(data, on=conv_id_key_name, how='inner')

    # @classmethod
    # def get_empathy_role_speaker(cls,
    #                              data: pd.DataFrame,
    #                              speaker_idx_key_name='speaker_idx',
    #                              conv_id_key_name='conv_id',
    #                              utter_idx_key_name='utterance_idx',
    #                              empathy_kind_key_name='empathy_kind',
    #                              result_key_name='empathy_role'):
    #     """
    #     get empathy_role of speaker
    #     (this function is written for detect which user can be empathetic bot and user)
    #     :param result_key_name:
    #     :param data:
    #     :param speaker_idx_key_name:
    #     :param conv_id_key_name:
    #     :param utter_idx_key_name:
    #     :param empathy_kind_key_name:
    #     :return:
    #     """
    #     conv_df = data[data[empathy_kind_key_name] == EmpathyKindEnum.PROVIDING.value]. \
    #         groupby([conv_id_key_name, speaker_idx_key_name, empathy_kind_key_name]).count().reset_index()
    #     conv_df = conv_df.loc[conv_df.reset_index().groupby([conv_id_key_name])[utter_idx_key_name].idxmax()]. \
    #         rename(columns={speaker_idx_key_name: result_key_name})
    #     return conv_df[[conv_id_key_name, result_key_name]].merge(data, on=conv_id_key_name, how='inner')

    @classmethod
    def segment_empathy_dialogue(cls,
                                 data: pd.pandas,
                                 conv_id_key_name='conv_id',
                                 empathy_kind_key_name='empathy_kind',
                                 new_conv_id_key_name='new_conv_id'):
        """
            برای سگمنت دو تا چیز مهمه یکی اینکه باید بری نوع همدلی و یکی همگام کنی این دوستان عزیز رو با اشخاصی که طرف روبه رو عه
            یعنی مکالمه از طرف یکی شروع میشه که جویای همدلیه و با یه فرد مقابل اولیه تموم میشه که همدلی رو تهیه میکنه یا کلا با شخص رو به رو واکنش همدلی نشون میده
            اینو به این مدلی که همدلی وجود داره یا نه هم باید بدی که ببینی همدلی وجود داره یا نه چون ممکنه نسبت به فرد مقابل و داده هاش همدلی نشون نده
            بعد یه تگ جدیدی میسازی که ببینی برای آیدی مکالمه جدید و ترتیب انجام گفتگو
            :param data:
            :param conv_id_key_name:
            :param empathy_kind_key_name:
            :param new_conv_id_key_name:
            :return:
        """
        conv_df = data.groupby(conv_id_key_name)[empathy_kind_key_name].apply(list).reset_index()
        conv_df[new_conv_id_key_name] = conv_df.\
            apply(lambda x: cls.get_new_conv_id_segments(empathy_kind_seq=x[empathy_kind_key_name],
                                                         cov_name_prefix=x[conv_id_key_name]),
                  axis=1)
        conv_df = conv_df.explode(new_conv_id_key_name)
        # todo filter the new conversations
        return conv_df[[conv_id_key_name, new_conv_id_key_name]].merge(data, on=conv_id_key_name, how='inner')

    @classmethod
    def get_new_conv_id_segments(cls,
                                 empathy_kind_seq: list,
                                 cov_name_prefix: str,
                                 default_conv_id=np.nan):
        """
            start with empathy_seeker and another speaker provide empathy
            conversation pattern "(12)*"
            and return list of new conv_id
            :param default_conv_id:
            :param cov_name_prefix:
            :param empathy_kind_seq:
            :return: list of new conv_ids
        """
        conv_id_positions = dict()
        empathy_kind_seq = "".join(empathy_kind_seq)
        for index, match_exp in enumerate(cls.EMPATHY_KIND_SEGMENT_CONDITION.finditer(empathy_kind_seq)):
            conv_id_positions.update({pos: f"{cov_name_prefix}_{index}" for pos in range(match_exp.start(),
                                                                                         match_exp.end())})
        return [conv_id_positions.get(i, default_conv_id) for i in range(len(empathy_kind_seq))]

