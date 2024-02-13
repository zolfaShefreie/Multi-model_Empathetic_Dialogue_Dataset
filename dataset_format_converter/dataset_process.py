import pandas as pd
import numpy as np
import ast
from abc import ABC, abstractmethod
import warnings
import os
import shutil

from utils import decorators
from settings import RAW_DATASET_PATH
from dataset_format_converter.conversation_utils import EmpathyFunctions, DialogueFunctions
from utils.decorators import WriterLoaderHandler, ChunkHandler
from utils.audio_utils import AudioModule
from utils.downloader import Downloader
from utils import other_utils
from utils.llms_utils import LLMsCompletionService


class BaseDialogueDatasetFormatter(ABC):
    """
    this class show base form of stages for convert multi-model dialogue dataset to empathetic multi-model
    dialogue dataset
    """

    # process configs
    DATASET_NAME = str()
    SEQ_STAGE = ['dataset_cleaner', 'audio_processing', 'filter_two_party', 'apply_empathy_classifier',
                 'filter_empathy_exist_conv', 'empathetic_segmentation', 'conversation_completion_checking',
                 'filter_missing_info', 'last_stage_changes']
    EDITABLE_STAGES = ['apply_empathy_classifier', 'conversation_completion_checking', ]
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = False
    NEED_VIDEO_TO_AUDIO = False
    NEED_AUDIO_SEGMENTATION = False
    AUDIO_FORMAT = 'wav'

    # metadata configs if metadata doesn't have these columns, these variables would use as default column name
    CONV_ID_COL_NAME = str()
    UTTER_ID_COL_NAME = str()
    UTTER_COL_NAME = str()
    SPEAKER_ID_COL_NAME = str()
    URL_COL_NAME = str()
    FILE_PATH_COL_NAME = str()

    NEW_CONV_ID_COL_NAME = "new_conv_id"
    NEW_UTTERANCE_IDX_NAME = "new_utter_idx"

    # Empathy Classifier using LLMs Configs
    USING_EMPATHY_CLASSIFIER_LLM = True
    EMPATHY_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    REQUEST_EACH_DATA_NUMBER = 10

    # Classifier for check completion of conversation using LLMs Configs
    USING_COMPLETE_CLASSIFIER_LLM = True
    COMPLETE_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    COMPLETE_REQUEST_EACH_DATA_NUMBER = 5
    IS_COMPLETE_COL_NAME = "is_complete"

    # if more columns change this list for dataset
    MAIN_COLUMNS = [CONV_ID_COL_NAME, UTTER_ID_COL_NAME, UTTER_COL_NAME, SPEAKER_ID_COL_NAME, FILE_PATH_COL_NAME]

    FILE_FORMAT = 'mp4'

    def __init__(self, dataset_dir: str, save_dir: str, chunk_length: int = None, *args, **kwargs):
        """
        initial of class
        :param dataset_dir: path of dataset
        :param save_dir: path for saving data after reformatting
        :param chunk_length: how many conversation do you want to run process for?
        :return:
        """
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        ChunkHandler.add_decorator_to_func(class_obj=self, dataset_name=self.DATASET_NAME,
                                           process_seq=self.SEQ_STAGE, group_by_keys=[self.CONV_ID_COL_NAME],
                                           chunk_length=chunk_length, data_arg_name='data')
        WriterLoaderHandler.add_decorator_to_func(class_obj=self, dataset_name=self.DATASET_NAME,
                                                  process_seq=self.SEQ_STAGE, editable_process=self.EDITABLE_STAGES)

    @abstractmethod
    def dataset_cleaner(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    # Audio Processing Module Part

    def audio_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        use each audio processing method in some conditions
        :param data: metadata with dataframe format
        :return: metadata with audio file path
        """
        if self.NEED_DOWNLOAD:
            data = self._download_manager(data)

        if self.NEED_VIDEO_TO_AUDIO:
            data = self._convertor_manager(data=data)

        if self.NEED_AUDIO_SEGMENTATION:
            data = self._audio_segmentation_manager(data=data)

        return data

    def _download_manager(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        download files from youtube with wav format and save in {dataset_dir}/audio_files/
        :param data: metadata with dataframe format
        :return: metadata with audio file path
        """
        # get unique urls for each conversation
        df = data[[self.CONV_ID_COL_NAME, self.URL_COL_NAME]].drop_duplicates()
        # download files and save files at dataset_dir
        df[self.FILE_PATH_COL_NAME] = df.apply(
            lambda x: Downloader.download(download_type='youtube',
                                          urls=[x[self.URL_COL_NAME], ],
                                          file_path=f"{self.dataset_dir}/audio_files/{x[self.CONV_ID_COL_NAME]}_{x.name}.{self.AUDIO_FORMAT}",
                                          file_format="wav"), axis=1)
        # merge the result with data
        return data.merge(df, on=[self.CONV_ID_COL_NAME, self.URL_COL_NAME], how='inner')

    def _convertor_manager(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        convert video to audio and save audio file at filepath location
        :param data: metadata with dataframe format
        :return: metadata with audio file path
        """
        print(data.columns)
        # get file paths of each conversation
        conv_df = data[[self.CONV_ID_COL_NAME, self.FILE_PATH_COL_NAME]].drop_duplicates()
        print(conv_df.columns)
        # create new column and save the path of audio
        # audio will be saved on video location with the same names
        conv_df[f"new_{self.FILE_PATH_COL_NAME}"] = conv_df.apply(
            lambda x: AudioModule.extract_audio_from_video(x[self.FILE_PATH_COL_NAME],
                                                           f"{x[self.FILE_PATH_COL_NAME].split(self.FILE_FORMAT)[0]}.{self.AUDIO_FORMAT}"),
            axis=1)

        # merge result with metadata and replace new column values with default file_path column
        return data.merge(conv_df, on=[self.CONV_ID_COL_NAME, self.FILE_PATH_COL_NAME], how='inner').\
            drop(columns=[self.FILE_PATH_COL_NAME]).\
            rename(columns={f"new_{self.FILE_PATH_COL_NAME}": self.FILE_PATH_COL_NAME})

    def _audio_segmentation_manager(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        segment audio files and save audio file for each utterance at filepath location
        :param data: metadata with dataframe format
        :return: metadata with audio file path for each utterance
        """
        # get unique path of audio of each conversation and the minimum turns
        conv_df = data.groupby([self.CONV_ID_COL_NAME, self.FILE_PATH_COL_NAME])[self.UTTER_ID_COL_NAME].apply(
            int).min().reset_index().rename(columns={self.UTTER_ID_COL_NAME: 'first_id'})
        # get list of utterance of each audio and merge to conv_df
        conv_df = conv_df.merge(
            data.groupby([self.CONV_ID_COL_NAME, self.FILE_PATH_COL_NAME])[self.UTTER_COL_NAME]).\
            apply(list).reset_index()
        # get list of utterance_ids
        conv_df = conv_df.merge(
            data.groupby([self.CONV_ID_COL_NAME, self.FILE_PATH_COL_NAME])[self.UTTER_ID_COL_NAME]).\
            apply(list).reset_index()
        # segment audio file
        conv_df['audio_files_path'] = conv_df.apply(
            lambda x: AudioModule.segment_audio(file_path=x[self.FILE_PATH_COL_NAME],
                                                utterances=x[self.UTTER_COL_NAME],
                                                prefix_name=x[self.CONV_ID_COL_NAME],
                                                save_dir=f"{'/'.join(x[self.FILE_PATH_COL_NAME].split('/')[:-1])}",
                                                first_utter_id=x['first_id']), axis=1)
        # explode the utter_ids and list of seg audio file path
        conv_df = conv_df.explode([self.UTTER_ID_COL_NAME, 'audio_files_path']).reset_index(drop=True)
        # merge result with metadata and replace new column values with default file_path column
        return data.merge(conv_df, on=[self.CONV_ID_COL_NAME, self.UTTER_ID_COL_NAME], how="inner").\
            drop(columns=[self.FILE_PATH_COL_NAME]).\
            rename(columns={'audio_files_path': self.FILE_PATH_COL_NAME})

    # Empathetic part

    @classmethod
    def filter_two_party(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        filter two part conversation based on class attribute
        :param data: metadata with dataframe format
        :return:
        """
        return DialogueFunctions.filter_two_party_dialogues(data=data,
                                                            conv_id_key_name=cls.CONV_ID_COL_NAME,
                                                            speaker_idx_key_name=cls.SPEAKER_ID_COL_NAME)

    # Empathy Module Part

    @classmethod
    def apply_empathy_classifier(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        using empathy classifiers to get empathy_kind and empathy_exist
        :param data: metadata with dataframe format
        :return:
        """
        if not cls.USING_EMPATHY_CLASSIFIER_LLM:
            return EmpathyFunctions.add_all_empathy_cols(data=data,
                                                         utter_key_name=cls.UTTER_COL_NAME,
                                                         utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                         conv_id_key_name=cls.CONV_ID_COL_NAME)
        else:
            return EmpathyFunctions.empathy_labels_using_llms(data=data,
                                                              utter_key_name=cls.UTTER_COL_NAME,
                                                              utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                              conv_id_key_name=cls.CONV_ID_COL_NAME,
                                                              tool=cls.EMPATHY_LLM_TOOL,
                                                              number_request=cls.REQUEST_EACH_DATA_NUMBER)

    @classmethod
    def filter_empathy_exist_conv(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        filter the empathetic conversation
        :param data: metadata with dataframe format
        :return:
        """
        if not cls.USING_EMPATHY_CLASSIFIER_LLM:
            return EmpathyFunctions.filter_empathetic_conversations(data=data,
                                                                    based_on='both',
                                                                    utter_key_name=cls.UTTER_COL_NAME,
                                                                    utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                                    conv_id_key_name=cls.CONV_ID_COL_NAME)

        else:
            return EmpathyFunctions.filter_empathetic_conversations(data=data,
                                                                    based_on='contain_empathy',
                                                                    utter_key_name=cls.UTTER_COL_NAME,
                                                                    utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                                    conv_id_key_name=cls.CONV_ID_COL_NAME)

    @classmethod
    def empathetic_segmentation(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        segment conversations to ge empathetic parts and adding missing info col for new conversations
        :param data: metadata with dataframe format
        :return:
        """
        data = EmpathyFunctions.segment_empathy_dialogue(data=data,
                                                         conv_id_key_name=cls.CONV_ID_COL_NAME,
                                                         utter_key_name=cls.UTTER_COL_NAME,
                                                         new_conv_id_key_name=cls.NEW_CONV_ID_COL_NAME,
                                                         new_utterance_id_key_name=cls.NEW_UTTERANCE_IDX_NAME)

        return data

    @classmethod
    def conversation_completion_checking(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        check each conversation that is complete or incomplete
        if you set it in configs to do it manually this function would add a new column
        :param data: metadata with dataframe format
        :return: data with new
        """
        if cls.USING_COMPLETE_CLASSIFIER_LLM:
            return DialogueFunctions.check_conv_is_complete_llms(data=data,
                                                                 utter_key_name=cls.UTTER_COL_NAME,
                                                                 utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                                 conv_id_key_name=cls.CONV_ID_COL_NAME,
                                                                 tool=cls.COMPLETE_LLM_TOOL,
                                                                 number_request=cls.COMPLETE_REQUEST_EACH_DATA_NUMBER,
                                                                 complete_key_name=cls.IS_COMPLETE_COL_NAME)
        else:
            # add missing information col for check conversations manually
            data[cls.IS_COMPLETE_COL_NAME] = 0
            return data

    @classmethod
    def filter_missing_info(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        filter missing_info == 1
        :param data: metadata with dataframe format
        :return: metadata without missing_info
        """
        warnings.warn(f"******************************************************************************\n"
                      f"WARNING: you must change {cls.IS_COMPLETE_COL_NAME} column in "
                      f"{WriterLoaderHandler.get_path(dataset_name=cls.DATASET_NAME, func_name='empathetic_segmentation', is_cache=False)}"
                      f" manually if you avoid using LLMs in previous stage to get correct result\n"
                      f"******************************************************************************")
        data[cls.IS_COMPLETE_COL_NAME] = data[cls.IS_COMPLETE_COL_NAME].apply(int)
        return data[data[cls.IS_COMPLETE_COL_NAME] == 1]

    def last_stage_changes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        delete some columns and rename some of them and save last changes
        :param data: metadata with dataframe format
        :return:
        """
        if self.NEW_UTTERANCE_IDX_NAME in data.columns:
            data = data.columns.drop(columns=[self.UTTER_ID_COL_NAME]).\
                rename(columns={self.NEW_UTTERANCE_IDX_NAME: self.UTTER_ID_COL_NAME})

        if self.NEW_CONV_ID_COL_NAME in data.columns:
            data = data.columns.drop(columns=[self.CONV_ID_COL_NAME]).\
                rename(columns={self.NEW_CONV_ID_COL_NAME: self.CONV_ID_COL_NAME})

        data = data[self.MAIN_COLUMNS]

        data = self._save_management(data=data)
        return data

    def _save_management(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        move audio file and save metadata on new dir
        :param data: metadata with dataframe format
        :return:
        """

        def move_file(save_dir, dataset_name, path, conv_id, utter_idx) -> str:
            """
            create new path and move audio file to it
            :param save_dir:
            :param dataset_name:
            :param path: path of audio
            :param conv_id: conversation id
            :param utter_idx: utterance id
            :return: new path
            """
            new_path = f"{save_dir}/{dataset_name}/audio_files/{conv_id}_{utter_idx}.{self.AUDIO_FORMAT}"
            os.rename(path, new_path)
            return new_path

        data[self.FILE_PATH_COL_NAME] = data.apply(lambda x: move_file(path=x[self.FILE_PATH_COL_NAME],
                                                                       save_dir=self.save_dir,
                                                                       dataset_name=self.DATASET_NAME,
                                                                       conv_id=x[self.CONV_ID_COL_NAME],
                                                                       utter_idx=x[self.UTTER_ID_COL_NAME]), axis=1)

        metadata_path = f"{self.save_dir}/{self.DATASET_NAME}/metadata.csv"
        data.to_csv(metadata_path)

        return data

    def running_process(self, start_stage: str = None, stop_stage: str = None):
        """
        run the set of stages between start stage and stop stage => [start_stage, stop_stage]
        :param start_stage: running stages form this stage
        :param stop_stage: stop running stages at this stage
        :return:
        """
        self._validate_stages(start_stage=start_stage, stop_stage=stop_stage)

        if not start_stage:
            start_stage_index = WriterLoaderHandler.get_process_stage(dataset_name=self.DATASET_NAME,
                                                                      process_seq=self.SEQ_STAGE)
            if start_stage_index >= len(self.SEQ_STAGE):
                return

        else:
            start_stage_index = self.SEQ_STAGE.index(start_stage)

        stop_stage_index = self.SEQ_STAGE.index(stop_stage) if stop_stage else len(self.SEQ_STAGE) - 1

        data = None
        for stage in self.SEQ_STAGE[start_stage_index: stop_stage_index + 1]:
            data = getattr(self, stage)(data=data)

    def _validate_stages(self, start_stage: str = None, stop_stage: str = None):
        """
        check start and stop stage be in stages list and the previous stage of start_stage be in cache
        :param start_stage: running stages form this stage
        :param stop_stage: stop running stages at this stage
        :return:
        """
        if start_stage and start_stage not in self.SEQ_STAGE:
            raise Exception("start_stage doesn't exists in stages")

        start_stage_index = WriterLoaderHandler.get_process_stage(dataset_name=self.DATASET_NAME,
                                                                  process_seq=self.SEQ_STAGE)

        # if the previous stage not be in .cache_stages
        if start_stage_index < self.SEQ_STAGE.index(start_stage):
            raise Exception("the previous stage doesn't run before")

        if stop_stage and stop_stage not in self.SEQ_STAGE:
            raise Exception("stop_stage doesn't exists in stages")

    @classmethod
    def stage_tracker(cls) -> list:
        """
        which stage running before
        :return: list of run stages
        """
        start_stage_index = WriterLoaderHandler.get_process_stage(dataset_name=cls.DATASET_NAME,
                                                                  process_seq=cls.SEQ_STAGE)
        # shows any stages of this dataset has not been run
        if start_stage_index == 0:
            return list()

        stages = cls.SEQ_STAGE[:start_stage_index]
        stages_info = [{'stage name': stages[0]}]
        for i in range(len(stages)-1):
            uncompleted, total = ChunkHandler.unprocessed_record_number(dataset_name=cls.DATASET_NAME,
                                                                        func_name=stages[i+1],
                                                                        pre_func_name=stages[i])
            stages_info.append({'stage name': stages[i+1],
                                'uncompleted records': {uncompleted},
                                'total records': total})

        return cls.SEQ_STAGE[:start_stage_index]


class AnnoMIDatasetFormatter(BaseDialogueDatasetFormatter):
    """
    this class is written based on csv data on AnnoMI github
    (https://github.com/uccollab/AnnoMI)
    """
    DATASET_NAME = 'AnnoMI'
    SEQ_STAGE = ['dataset_cleaner', 'audio_processing', 'filter_two_party', 'apply_empathy_classifier',
                 'filter_empathy_exist_conv', 'empathetic_segmentation', 'filter_missing_info', 'last_stage_changes']
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = True
    NEED_VIDEO_TO_AUDIO = False
    NEED_AUDIO_SEGMENTATION = True
    AUDIO_FORMAT = 'wav'

    # metadata configs if metadata doesn't have these columns, these variables would use as default column name
    CONV_ID_COL_NAME = "transcript_id"
    UTTER_ID_COL_NAME = "utterance_id"
    UTTER_COL_NAME = "utterance_text"
    SPEAKER_ID_COL_NAME = "interlocutor"
    URL_COL_NAME = "video_url"
    FILE_PATH_COL_NAME = "file_path"

    NEW_CONV_ID_COL_NAME = "new_conv_id"
    NEW_UTTERANCE_IDX_NAME = "new_utter_idx"

    # Empathy Classifier using LLMs Configs
    USING_EMPATHY_CLASSIFIER_LLM = True
    EMPATHY_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    REQUEST_EACH_DATA_NUMBER = 10

    # Classifier for check completion of conversation using LLMs Configs
    USING_COMPLETE_CLASSIFIER_LLM = True
    COMPLETE_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    COMPLETE_REQUEST_EACH_DATA_NUMBER = 5
    IS_COMPLETE_COL_NAME = "is_complete"

    # if more columns change this list for dataset
    MAIN_COLUMNS = [CONV_ID_COL_NAME, UTTER_ID_COL_NAME, UTTER_COL_NAME, SPEAKER_ID_COL_NAME, FILE_PATH_COL_NAME,
                    'client_talk_type', 'main_therapist_behaviour']

    FILE_FORMAT = 'wav'

    def dataset_cleaner(self, *args, **kwargs) -> pd.DataFrame:
        """
        convert raw dataset the special format
        :param args:
        :param kwargs:
        :return:
        """
        return pd.read_csv(f"{self.dataset_dir}/AnnoMI-full.csv")


class DailyTalkDatasetFormatter(BaseDialogueDatasetFormatter):
    """
    This class is written based on data on google drive of dailyTalk
    (https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL)
    """

    # process configs
    DATASET_NAME = 'dailyTalk'
    SEQ_STAGE = ['dataset_cleaner', 'audio_processing', 'filter_two_party', 'apply_empathy_classifier',
                 'filter_empathy_exist_conv', 'empathetic_segmentation', 'filter_missing_info', 'last_stage_changes']
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = False
    NEED_VIDEO_TO_AUDIO = False
    NEED_AUDIO_SEGMENTATION = False
    AUDIO_FORMAT = 'wav'

    # metadata configs if metadata doesn't have these columns, these variables would use as default column name
    CONV_ID_COL_NAME = "dialog_idx"
    UTTER_ID_COL_NAME = "utterance_idx"
    UTTER_COL_NAME = "text"
    SPEAKER_ID_COL_NAME = "speaker"
    URL_COL_NAME = None
    FILE_PATH_COL_NAME = "file_path"

    NEW_CONV_ID_COL_NAME = "new_conv_id"
    NEW_UTTERANCE_IDX_NAME = "new_utter_idx"

    # Empathy Classifier using LLMs Configs
    USING_EMPATHY_CLASSIFIER_LLM = True
    EMPATHY_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    REQUEST_EACH_DATA_NUMBER = 10

    # Classifier for check completion of conversation using LLMs Configs
    USING_COMPLETE_CLASSIFIER_LLM = True
    COMPLETE_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    COMPLETE_REQUEST_EACH_DATA_NUMBER = 5
    IS_COMPLETE_COL_NAME = "is_complete"

    # if more columns change this list for dataset
    MAIN_COLUMNS = [CONV_ID_COL_NAME, UTTER_ID_COL_NAME, UTTER_COL_NAME, SPEAKER_ID_COL_NAME, FILE_PATH_COL_NAME,
                    'emotion', 'act']

    FILE_FORMAT = 'wav'

    def dataset_cleaner(self, *args, **kwargs) -> pd.DataFrame:
        """
        convert raw dataset the special format
        :param args:
        :param kwargs:
        :return:
        """
        metadata_file_path = f"{self.dataset_dir}/metadata.json"
        data = self._convert_metadata_to_dataframe(metadata_path=metadata_file_path)
        data = self._add_audio_file_path_col(data=data)
        return data

    @staticmethod
    def _get_folder_raw_data(raw_dataset_path):
        if raw_dataset_path.endswith('.zip'):
            return other_utils.unzip(raw_dataset_path, RAW_DATASET_PATH + 'DialyDialoge_DailyTalk')
        return raw_dataset_path

    def _convert_metadata_to_dataframe(self, metadata_path: str) -> pd.DataFrame:
        """
        covert json to pd.Dataframe
        :param metadata_path: path of metadata josn file
        :return: metadata with dataframe format
        """
        with open(metadata_path, encoding="utf8") as file:
            data = ast.literal_eval(file.read())
            return pd.DataFrame([utterance_data for conversations in data.values()
                                for utterance_data in conversations.values()])

    def _add_audio_file_path_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add file_path col to metadata
        :param data: metadata with dataframe format
        :return: metadata with file_path_col
        """

        def get_audio_file_path(dataset_dir, conv_id, utter_id, speaker_id):
            """
            get path of audio file based on this dataset
            :param dataset_dir:
            :param conv_id:
            :param utter_id:
            :param speaker_id:
            :return:
            """
            path = f"{dataset_dir}/data/{conv_id}/{utter_id}_{speaker_id}_d{conv_id}.wav"
            return path if os.path.exists(path) else None

        data[self.FILE_PATH_COL_NAME] = data.apply(
            lambda x: get_audio_file_path(dataset_dir=self.dataset_dir,
                                          conv_id=x[self.CONV_ID_COL_NAME],
                                          utter_id=x[self.UTTER_ID_COL_NAME],
                                          speaker_id=x[self.SPEAKER_ID_COL_NAME]), axis=1)
        return data


class MELDDatasetFormatter(BaseDialogueDatasetFormatter):
    """
    This class is written based on data on below link
    (https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
    but use the two party conversations files in github of meld instead of csv files in raw meld dir
    (https://github.com/declare-lab/MELD/tree/master/data/MELD_Dyadic)
    """

    # process configs
    DATASET_NAME = 'MELD'
    SEQ_STAGE = ['dataset_cleaner', 'audio_processing', 'filter_two_party', 'apply_empathy_classifier',
                 'filter_empathy_exist_conv', 'empathetic_segmentation', 'filter_missing_info', 'last_stage_changes']
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = False
    NEED_VIDEO_TO_AUDIO = True
    NEED_AUDIO_SEGMENTATION = False
    AUDIO_FORMAT = 'wav'

    # metadata configs if metadata doesn't have these columns, these variables would use as default column name
    CONV_ID_COL_NAME = "Dialogue_ID"
    UTTER_ID_COL_NAME = "Utterance_ID"
    UTTER_COL_NAME = "Utterance"
    SPEAKER_ID_COL_NAME = "Speaker"
    URL_COL_NAME = None
    FILE_PATH_COL_NAME = "file_path"

    NEW_CONV_ID_COL_NAME = "new_conv_id"
    NEW_UTTERANCE_IDX_NAME = "new_utter_idx"

    SPLIT_COL_NAME = 'split'
    FILE_CONV_ID = 'Old_Dialogue_ID'
    FILE_UTTER_ID = 'Old_Utterance_ID'

    # Empathy Classifier using LLMs Configs
    USING_EMPATHY_CLASSIFIER_LLM = True
    EMPATHY_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    REQUEST_EACH_DATA_NUMBER = 10

    # Classifier for check completion of conversation using LLMs Configs
    USING_COMPLETE_CLASSIFIER_LLM = True
    COMPLETE_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    COMPLETE_REQUEST_EACH_DATA_NUMBER = 5
    IS_COMPLETE_COL_NAME = "is_complete"

    # if more columns change this list for dataset
    MAIN_COLUMNS = [CONV_ID_COL_NAME, UTTER_ID_COL_NAME, UTTER_COL_NAME, SPEAKER_ID_COL_NAME, FILE_PATH_COL_NAME,
                    'Emotion', 'Sentiment']

    FILE_FORMAT = 'mp4'

    def dataset_cleaner(self, *args, **kwargs) -> pd.DataFrame:
        """
        convert raw dataset the special format
        :param args:
        :param kwargs:
        :return:
        """
        # merge dataframe for each split
        data = self._append_multi_dataframe({'dev': pd.read_csv(f"{self.dataset_dir}/dev_sent_emo_dya.csv"),
                                            'test': pd.read_csv(f"{self.dataset_dir}/test_sent_emo_dya.csv"),
                                             'train': pd.read_csv(f"{self.dataset_dir}/train_sent_emo_dya.csv"), })

        # filter the dialogue with continuous multi turn for one party
        data = DialogueFunctions.filter_not_multi_turn_on_one_party(data=data,
                                                                    conv_id_key_name=self.CONV_ID_COL_NAME,
                                                                    utter_id_key_name=self.UTTER_ID_COL_NAME,
                                                                    speaker_id_key_name=self.SPEAKER_ID_COL_NAME)
        # file managing
        return self._add_file_path_col(data)

    @classmethod
    def _append_multi_dataframe(cls, data_dict: dict) -> pd.DataFrame:
        """
        change the conv_id and append test, train, dev dataframe
        :param data_dict: dict of dataframe with split name as key
        :return: one appended dataframe
        """
        for key, data in data_dict.items():
            data[cls.SPLIT_COL_NAME] = key

        data_list = list(data_dict.values())
        data = data_list[0]
        for df in data_list[1:]:
            # df[cls.CONV_ID_COL_NAME] = data[cls.CONV_ID_COL_NAME].apply(np.int64)
            df[cls.CONV_ID_COL_NAME] = df[cls.CONV_ID_COL_NAME] + len(data)
            data = pd.concat([data, df], ignore_index=True)
        return data

    def _add_file_path_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        copy file to new dir (different utterances have same files) and add file_path col
        :param data: metadata with dataframe format
        :return:
        """

        path_metadata_dict = {
            'train': 'train/train_splits',
            'dev': 'dev/dev_splits_complete',
            'test': 'test/output_repeated_splits_test'
        }

        def copy_rename_file(row):
            """
            copy the file with new name for each utterance
            :param row:
            :return:
            """
            old_path = f"{self.dataset_dir}/{path_metadata_dict[row[self.SPLIT_COL_NAME]]}/" \
                       f"dia{row[self.FILE_CONV_ID]}_utt{row[self.FILE_UTTER_ID]}.mp4"

            new_path = f"{self.dataset_dir}/audio_files/{row[self.CONV_ID_COL_NAME]}_{row[self.UTTER_ID_COL_NAME]}.mp3"
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy2(src=old_path, dst=new_path)
            return new_path

        data[self.FILE_PATH_COL_NAME] = data.apply(copy_rename_file, axis=1)
        return data


class MUStARDDatasetFormatter(BaseDialogueDatasetFormatter):
    """
       This class is written based on data on google drive of MUStARD
       (https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view)
       json file from
       (https://github.com/soujanyaporia/MUStARD/blob/master/data/sarcasm_data.json)
       """

    # process configs
    DATASET_NAME = 'MUStARD'
    SEQ_STAGE = ['dataset_cleaner', 'audio_processing', 'filter_two_party', 'apply_empathy_classifier',
                 'filter_empathy_exist_conv', 'empathetic_segmentation', 'filter_missing_info', 'last_stage_changes']
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = False
    NEED_VIDEO_TO_AUDIO = True
    NEED_AUDIO_SEGMENTATION = True
    AUDIO_FORMAT = 'wav'

    # metadata configs if metadata doesn't have these columns, these variables would use as default column name
    CONV_ID_COL_NAME = "conv_id"
    UTTER_ID_COL_NAME = "utter_id"
    UTTER_COL_NAME = "utterance"
    SPEAKER_ID_COL_NAME = "speaker"
    URL_COL_NAME = None
    FILE_PATH_COL_NAME = "file_path"

    NEW_CONV_ID_COL_NAME = "new_conv_id"
    NEW_UTTERANCE_IDX_NAME = "new_utter_idx"

    TYPE_UTTER_COL_NAME = 'type_utter'
    RECORD_ID_COL_NAME = 'record_id'

    # Empathy Classifier using LLMs Configs
    USING_EMPATHY_CLASSIFIER_LLM = True
    EMPATHY_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    REQUEST_EACH_DATA_NUMBER = 10

    # Classifier for check completion of conversation using LLMs Configs
    USING_COMPLETE_CLASSIFIER_LLM = True
    COMPLETE_LLM_TOOL = LLMsCompletionService.Tools.OPENAI
    COMPLETE_REQUEST_EACH_DATA_NUMBER = 5
    IS_COMPLETE_COL_NAME = "is_complete"

    # if more columns change this list for dataset
    MAIN_COLUMNS = [CONV_ID_COL_NAME, UTTER_ID_COL_NAME, UTTER_COL_NAME, SPEAKER_ID_COL_NAME, FILE_PATH_COL_NAME,
                    'scam']

    FILE_FORMAT = 'mp4'

    def dataset_cleaner(self, *args, **kwargs) -> pd.DataFrame:
        """
        convert raw dataset the special format
        :param args:
        :param kwargs:
        :return:
        """
        data = self._convert_metadata_to_dataframe(f"{self.dataset_dir}/sarcasm_data.json")
        return self._add_audio_file_path_col(data=data)

    def _convert_metadata_to_dataframe(self, metadata_path: str) -> pd.DataFrame:
        """
        covert json to pd.Dataframe
        :param metadata_path: path of metadata josn file
        :return: metadata with dataframe format
        """
        with open(metadata_path, encoding="utf8") as file:
            dict_content = ast.literal_eval(file.read())
            data = list()
            for conv_id, item in enumerate(dict_content.items()):
                record_id, record = item
                # source content
                for utter_id in range(len(record['context'])):
                    utter = record['context'][utter_id]
                    speaker = record['context_speakers'][utter_id]
                    data.append({self.CONV_ID_COL_NAME: conv_id,
                                 self.UTTER_ID_COL_NAME: utter_id,
                                 self.UTTER_COL_NAME: utter,
                                 self.SPEAKER_ID_COL_NAME: speaker,
                                 'scam': record['sarcasm'],
                                 self.TYPE_UTTER_COL_NAME: 'context',
                                 self.RECORD_ID_COL_NAME: record_id})
                # target
                data.append({self.CONV_ID_COL_NAME: conv_id,
                             self.UTTER_ID_COL_NAME: len(record['context']),
                             self.UTTER_COL_NAME: record['utterance'],
                             self.SPEAKER_ID_COL_NAME: record['speaker'],
                             'scam': record['sarcasm'],
                             self.TYPE_UTTER_COL_NAME: 'target',
                             self.RECORD_ID_COL_NAME: record_id})

            return pd.DataFrame(data)

    def _add_audio_file_path_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add file_path col to metadata
        :param data: metadata with dataframe format
        :return: metadata with file_path_col
        """

        def get_audio_file_path(dataset_dir, record_id, type_utter):
            """
            get path of audio file based on this dataset
            :param dataset_dir:
            :param record_id:
            :param type_utter:
            :return:
            """

            path = f"{dataset_dir}/{'context_final' if type_utter=='context' else 'utterances_final'}" \
                   f"/{record_id}{'_c' if type_utter=='context' else ''}.mp4"
            return path if os.path.exists(path) else None

        data[self.FILE_PATH_COL_NAME] = data.apply(
            lambda x: get_audio_file_path(dataset_dir=self.dataset_dir,
                                          record_id=x[self.RECORD_ID_COL_NAME],
                                          type_utter=x[self.TYPE_UTTER_COL_NAME]), axis=1)
        return data
