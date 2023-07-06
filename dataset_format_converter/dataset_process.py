import pandas as pd
import json
import ast
from abc import ABC, abstractmethod

from utils import other_utils
from settings import RAW_DATASET_PATH
from dataset_format_converter.conversation_utils import EmpathyFunctions, DialogueFunctions
from utils.other_utils import WriterLoaderHandler


class BaseDialogueDatasetFormatter(ABC):
    """
    this class show base form of stages for convert multi-model dialogue dataset to empathetic multi-model
    dialogue dataset
    """

    # process configs
    DATASET_NAME = str()
    SEQ_STAGE = []
    # some audio or video files were uploaded on youtube
    NEED_DOWNLOAD = False
    NEED_VIDEO_TO_AUDIO = False
    NEED_AUDIO_SEGMENTATION = False

    # metadata configs
    CONV_ID_COL_NAME = str()
    UTTER_ID_COL_NAME = str()
    UTTER_COL_NAME = str()
    SPEAKER_ID_COL_NAME = str()

    def __int__(self, dataset_dir: str, save_dir: str):
        """
        initial of class
        :param dataset_dir: path of dataset
        :param save_dir: path for saving data after reformatting
        :return:
        """
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir

    @abstractmethod
    def dataset_cleaner(self):
        raise NotImplementedError

    # Audio Processing Module Part

    @abstractmethod
    def file_path_manager(self):
        raise NotImplementedError

    def audio_processing(self):
        if self.NEED_DOWNLOAD:
            pass

        if self.NEED_VIDEO_TO_AUDIO:
            pass

        if self.NEED_AUDIO_SEGMENTATION:
            pass

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
    @WriterLoaderHandler.decorator(dataset_name=DATASET_NAME, process_seq=SEQ_STAGE, human_editable=True)
    def apply_empathy_classifier(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        using empathy classifiers to get empathy_kind and empathy_exist
        :param data: metadata with dataframe format
        :return:
        """
        return EmpathyFunctions.add_all_empathy_cols(data=data,
                                                     utter_key_name=cls.UTTER_COL_NAME,
                                                     utter_id_key_name=cls.UTTER_ID_COL_NAME,
                                                     conv_id_key_name=cls.CONV_ID_COL_NAME)

    def filter_empathy_exist_conv(self, data):
        # todo: EmpathyFunctions.filter_empathetic_conversations
        pass

    @WriterLoaderHandler.decorator(dataset_name=DATASET_NAME, process_seq=SEQ_STAGE, human_editable=True)
    def empathetic_segmentation(self):
        pass

    def running_process(self, start_stage: str = None, stop_stage: str = None):
        pass


class AnnoMIDatasetFormatter:
    """
    this class is written based on csv data on AnnoMI github
    (https://github.com/uccollab/AnnoMI)
    """
    pass


class DailyTalkDatasetFormatter:
    """
    This class is written based on data on google drive of dailyTalk
    (https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL)
    """
    # todo: 2. بری فایل متادیتاش رو برداری بهتره با دیکشنری جلو بری تبدیل کنی به اون فرمت دیتا فریم
    # todo 3, فرمت فایل ها رو باید درست کنی همه تو یه پوشه فقط یه مسیر تو دیتافریم اضافه میشه که مسیر به اون گفته ست
    # todo: 4. برای اون فانکشن‌های همدلی ببینی اینا همدلانه هست یا نه
    # todo: 5, البته می تونی سگمت هم کنی که با یه چیز جویای همدلی شروع بشه و کجا مکالمه تموم بشه
    # todo: 6, اگه خواستی دستی تغییر بده اینا رو
    # todo: 7, یه فیلتر بزن و اونایی که نیاز داری رو ببر تو یه مسیر جدا و قبلی یا رو پاک کن

    def __init__(self, raw_dataset_path: str):
        self.folder = "./dailytalk/"
        self.json_file = f"{self.folder}/metadata.json"
        self._get_folder_raw_data(raw_dataset_path=raw_dataset_path)

    @staticmethod
    def _get_folder_raw_data(raw_dataset_path):
        if raw_dataset_path.endswith('.zip'):
            return other_utils.unzip(raw_dataset_path, RAW_DATASET_PATH + 'DialyDialoge_DailyTalk')
        return raw_dataset_path

    def _convert_metadata_to_dataframe(self) -> pd.DataFrame:
        with open(self.json_file) as file:
            data = ast.literal_eval(file.read())
            return pd.DataFrame([utterance_data for conversations in data.values()
                                for utterance_data in conversations.values()])

    # todo:complete
    def _add_audio_file_path_col(self, data: pd.DataFrame):
        pass

    def _add_empathy_cols(self, data: pd.DataFrame):
        return EmpathyFunctions.add_all_empathy_cols(data=data,
                                                     utter_key_name='text',
                                                     utter_id_key_name='utterance_idx',
                                                     conv_id_key_name='dialog_idx')



# class DatasetFormatter:
#
#     @classmethod
#     def annoml_formatter(cls, raw_dataset_csv_path):
#         pass
#
#     @classmethod
#     def dailytalk_formatter(cls, raw_dataset_zip_path):
#         des_path = other_utils.unzip(raw_dataset_zip_path, RAW_DATASET_PATH + 'DialyDialoge_DailyTalk')

