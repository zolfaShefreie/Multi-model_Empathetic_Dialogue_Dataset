import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import re
import enum

from base_models_classes.empathy_kind import EmpathyKindClassifier, EmpathyKindEnum, EmpathyKindClassifierLLMs
from base_models_classes.exist_empathy import ExistEmpathyClassifier
from base_models_classes.complete_checker import CompleteCheckerClassifierLLMs


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

    @classmethod
    def make_utter_id_seq(cls,
                          data: pd.DataFrame,
                          conv_id_key_name='conv_id',
                          utter_key_name='utterance',
                          result_key_name='utterance_idx'):
        """
        make a sequence number for utterance idx
        :param data:
        :param conv_id_key_name: name of conv_id column
        :param utter_key_name: name of utterance column
        :param result_key_name: name of utterance_idx column
        :return:
        """
        conv_df = data[[conv_id_key_name, utter_key_name]].groupby([conv_id_key_name])[utter_key_name].\
            apply(list).reset_index()
        conv_df[result_key_name] = conv_df.apply(lambda x: [i for i, value in enumerate(x[utter_key_name])], axis=1)
        conv_df = conv_df.explode([utter_key_name, result_key_name])
        return data.merge(conv_df, on=[conv_id_key_name, utter_key_name], how='inner')

    @classmethod
    def check_conv_is_complete_llms(cls,
                                    data: pd.DataFrame,
                                    number_request: int,
                                    tool: enum.Enum,
                                    utter_key_name='utterance',
                                    conv_id_key_name='conv_id',
                                    utter_id_key_name='utterance_idx',
                                    complete_key_name: str = 'is_completed',
                                    reasons_key_name: str = 'complete_check_reasons',
                                    percent_key_name: str = 'complete_check_percent'):
        """
        management of whole dataframe to get label results from LLMs
        :param data: dataframe of conversations
        :param number_request: number of requests for each conversation
        :param tool: which tool do you want to use for completion task?
        :param utter_key_name: name of col that have utterance values
        :param utter_id_key_name: name of col that have utterance_idx values
        :param conv_id_key_name: name of col that have conv_id values
        :param complete_key_name: the key name of complete_check_label. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param percent_key_name: the key name of percents. use for result
        :return:
        """

        def get_labels(utterances: list,
                       number_request: int,
                       tool: enum.Enum,
                       complete_key_name: str = 'is_completed',
                       reasons_key_name: str = 'complete_check_reasons',
                       percent_key_name: str = 'complete_check_percent') -> list:
            """
            get the string of conversation and get the label and other info for each utterance using LLMs
            :param utterances: list of utterance
            :param number_request: number of requests for each conversation
            :param tool: which tool do you want to use for completion task?
            :param complete_key_name: the key name of complete_check_label. use for result
            :param reasons_key_name: the key name of reasons. use for result
            :param percent_key_name: the key name of percents. use for result
            :return:
            """

            conv_str = cls.str_conv_prompt_format(utterances=utterances)
            result = CompleteCheckerClassifierLLMs.run_process(conv_str=conv_str,
                                                               number_request=number_request,
                                                               tool=tool,
                                                               complete_key_name=complete_key_name,
                                                               reasons_key_name=reasons_key_name,
                                                               percent_key_name=percent_key_name)
            return result

        # sort based on utterance turn
        data = data.sort_values(by=[conv_id_key_name, utter_id_key_name])
        conv_df = data[[conv_id_key_name, utter_key_name]].groupby([conv_id_key_name])[utter_key_name].apply(list) \
            .reset_index()

        # get the result
        conv_df['llm_result'] = conv_df.apply(lambda x: get_labels(utterances=x[utter_key_name],
                                                                   number_request=number_request,
                                                                   tool=tool,
                                                                   complete_key_name=complete_key_name,
                                                                   reasons_key_name=reasons_key_name,
                                                                   percent_key_name=percent_key_name), axis=1)
        # make new columns for result of LLMs
        conv_df = conv_df.merge(conv_df['llm_result'].apply(pd.Series), left_index=True, right_index=True)
        # merge to original data
        return data.merge(conv_df[[conv_id_key_name, complete_key_name,
                                   reasons_key_name, percent_key_name]],
                          on=[conv_id_key_name, utter_id_key_name], how='inner')

    @classmethod
    def filter_not_multi_turn_on_one_party(cls,
                                           data: pd.DataFrame,
                                           conv_id_key_name='conv_id',
                                           utter_id_key_name='utterance_idx',
                                           speaker_id_key_name='speaker_idx') -> pd.DataFrame:
        """
        filter the conversations that one party doesn't get multi turn continuously
        :param data:
        :param conv_id_key_name: name of conv_id column
        :param utter_id_key_name: name of utterance column
        :param speaker_id_key_name: name of speaker_idx column
        :return:
        """

        def is_multi_turn_one_party(row) -> bool:
            """
            check the sequence of turns it (speaker_1, speaker_2)*
            :param row:
            :return: a bool
            """
            speaker_list = row[speaker_id_key_name]
            speaker_1, speaker_2 = speaker_list[0], speaker_list[1]
            if speaker_1 == speaker_2:
                return False

            for index, speaker_id in enumerate(speaker_list):
                if index % 2 == 0 and speaker_1 != speaker_id:
                    return False

                if index % 2 == 1 and speaker_2 != speaker_id:
                    return False

            return True

        data = data.sort_values([conv_id_key_name, utter_id_key_name])
        conv_df = data.groupby([conv_id_key_name])[speaker_id_key_name].apply(list).reset_index()
        conv_df = conv_df[conv_df.apply(is_multi_turn_one_party, axis=1)]
        return data.merge(conv_df[[conv_id_key_name]], on=[conv_id_key_name], how='inner')

    @classmethod
    def str_conv_prompt_format(cls, utterances: list) -> str:
        """
        return str of conversation based on empathy prompt format
        :param utterances: sorted utterances based on turn
        :return: str of conversation
        """
        conv_str = str()
        for i, utter in enumerate(utterances):
            conv_str += f"{i + 1}. {utter}\n"
        return conv_str


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
    def empathy_labels_using_llms(cls,
                                  data: pd.DataFrame,
                                  number_request: int,
                                  tool: enum.Enum,
                                  utter_key_name='utterance',
                                  utter_id_key_name='utterance_idx',
                                  conv_id_key_name='conv_id',
                                  empathy_key_name: str = 'empathy_kind',
                                  reasons_key_name: str = 'empathy_reasons',
                                  percent_key_name: str = 'empathy_percents') -> pd.DataFrame:

        """
        management of whole dataframe to get label results from LLMs
        :param data: dataframe of conversations
        :param number_request: number of requests for each conversation
        :param tool: which tool do you want to use for completion task?
        :param utter_key_name: name of col that have utterance values
        :param utter_id_key_name: name of col that have utterance_idx values
        :param conv_id_key_name: name of col that have conv_id values
        :param empathy_key_name: the key name of empathy_label. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param percent_key_name: the key name of percents. use for result
        :return:
        """

        def get_labels(utterances: list,
                       number_request: int,
                       tool: enum.Enum,
                       empathy_key_name: str = 'Empathy',
                       reasons_key_name: str = 'empathy_reasons',
                       percent_key_name: str = 'empathy_percents') -> list:
            """
            get the string of conversation and get the label and other info for each utterance using LLMs
            :param utterances: list of utterance
            :param number_request: number of requests for each conversation
            :param tool: which tool do you want to use for completion task?
            :param empathy_key_name: the key name of empathy_label. use for result
            :param reasons_key_name: the key name of reasons. use for result
            :param percent_key_name: the key name of percents. use for result
            :return:
            """

            conv_str = DialogueFunctions.str_conv_prompt_format(utterances=utterances)
            result = EmpathyKindClassifierLLMs.run_process(conv_str=conv_str,
                                                           number_of_utter=len(utterances),
                                                           number_request=number_request,
                                                           tool=tool,
                                                           empathy_key_name=empathy_key_name,
                                                           reasons_key_name=reasons_key_name,
                                                           percent_key_name=percent_key_name)
            return result

        # sort based on utterance turn
        data = data.sort_values(by=[conv_id_key_name, utter_id_key_name])
        # group by conv and make list of utterances and their ids
        conv_df = data[[conv_id_key_name, utter_key_name]].groupby([conv_id_key_name])[utter_key_name].apply(list)\
            .reset_index()
        conv_df = conv_df.merge(data[[conv_id_key_name, utter_id_key_name]].groupby([conv_id_key_name])[utter_id_key_name]
                                .apply(list).reset_index(), how='inner', on=[conv_id_key_name])

        # get the result
        conv_df['em_llm_result'] = conv_df.apply(lambda x: get_labels(utterances=x[utter_key_name],
                                                                      number_request=number_request,
                                                                      tool=tool,
                                                                      empathy_key_name=empathy_key_name,
                                                                      reasons_key_name=reasons_key_name,
                                                                      percent_key_name=percent_key_name), axis=1)

        # explode to get info for each utterance in dataframe as a row
        conv_df = conv_df.explode([utter_id_key_name, utter_key_name, 'em_llm_result']).reset_index(drop=True)
        # make new columns for result of LLMs
        conv_df = conv_df.merge(conv_df['em_llm_result'].apply(pd.Series), left_index=True, right_index=True)
        # merge to original data
        return data.merge(conv_df[[conv_id_key_name, utter_id_key_name, empathy_key_name,
                                   reasons_key_name, percent_key_name]],
                          on=[conv_id_key_name, utter_id_key_name], how='inner')

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
                                        is_empathy_key_name='is_empathy',
                                        utter_key_name='utterance',
                                        utter_id_key_name='utterance_idx',
                                        conv_id_key_name='conv_id',
                                        empathy_kind_key_name='empathy_kind',
                                        empathy_seq_key_name='empathy_kind_seq'):
        """
        filter conversations based on empathy_kind sequence and empathy existence
        :param empathy_seq_key_name:
        :param empathy_kind_key_name:
        :param conv_id_key_name:
        :param utter_id_key_name:
        :param utter_key_name:
        :param data:
        :param based_on: can be 'both', 'contain_empathy', 'is_empathy'
        :param contain_empathy_key_name:
        :param is_empathy_key_name:
        :return:
        """
        if (based_on in ['both', 'contain_empathy']) and \
                ((empathy_seq_key_name not in data.columns) or (contain_empathy_key_name not in data.columns)):
            data = cls.check_empathy_kind_seq(data=data,
                                              utter_key_name=utter_key_name,
                                              utter_id_key_name=utter_id_key_name,
                                              conv_id_key_name=conv_id_key_name,
                                              empathy_kind_key_name=empathy_kind_key_name,
                                              empathy_seq_key_name=empathy_seq_key_name,
                                              result_key_name=contain_empathy_key_name)

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
                                 utter_key_name='utterance',
                                 utter_id_key_name='utterance_idx',
                                 empathy_kind_key_name='empathy_kind',
                                 new_conv_id_key_name='new_conv_id',
                                 new_utterance_id_key_name='new_utterance_id'):
        """
        management of segmentation
            :param utter_id_key_name:
            :param data:
            :param conv_id_key_name:
            :param empathy_kind_key_name:
            :param utter_key_name:
            :param new_conv_id_key_name:
            :param new_utterance_id_key_name:
            :return:
        """
        # group by conv and make list of utterances_ids and their empathy_kind
        conv_df = data[[conv_id_key_name, utter_id_key_name]].groupby([conv_id_key_name])[utter_id_key_name].apply(list) \
            .reset_index()
        conv_df = conv_df.merge(
            data[[conv_id_key_name, empathy_kind_key_name]].groupby([conv_id_key_name])[empathy_kind_key_name]
            .apply(list).reset_index(), how='inner', on=[conv_id_key_name])
        # get new conv_id for each segment
        conv_df[new_conv_id_key_name] = conv_df.\
            apply(lambda x: cls.get_new_conv_id_segments(empathy_kind_seq=x[empathy_kind_key_name],
                                                         cov_name_prefix=x[conv_id_key_name]),
                  axis=1)
        # explode two column list
        conv_df = conv_df.explode([utter_id_key_name, new_conv_id_key_name])
        # get new conversations
        new_data = conv_df[[conv_id_key_name, new_conv_id_key_name, utter_id_key_name]].\
            merge(data, on=[conv_id_key_name, utter_id_key_name], how='inner')
        return DialogueFunctions.make_utter_id_seq(data=new_data[new_data[new_conv_id_key_name].notnull()],
                                                   conv_id_key_name=new_conv_id_key_name,
                                                   utter_key_name=utter_key_name,
                                                   result_key_name=new_utterance_id_key_name)

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
        # convert list of empathy_kind to one string
        empathy_kind_seq = "".join([str(em_kind) for em_kind in empathy_kind_seq])
        for index, match_exp in enumerate(cls.EMPATHY_KIND_SEGMENT_CONDITION.finditer(empathy_kind_seq)):
            conv_id_positions.update({pos: f"{cov_name_prefix}_{index}" for pos in range(match_exp.start(),
                                                                                         match_exp.end())})
        return [conv_id_positions.get(i, default_conv_id) for i in range(len(empathy_kind_seq))]
