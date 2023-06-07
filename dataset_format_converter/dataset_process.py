import pandas as pd
import json
import ast

from utils import other_utils
from settings import RAW_DATASET_PATH
from dataset_format_converter.general_utils import EmpathyFunctions


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

