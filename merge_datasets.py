import os
import shutil
import pandas as pd
from datasets import load_dataset

from dataset_format_converter.dataset_process import MELDDatasetFormatter, DailyTalkDatasetFormatter, \
    MUStARDDatasetFormatter, AnnoMIDatasetFormatter
from settings import HUGGING_FACE_REPO_NAME, HUGGING_FACE_IS_PRIVATE, HUGGING_FACE_TOKEN


class DatasetMerger:
    DATASETS = {
        'meld': MELDDatasetFormatter,
        'dailytalk': DailyTalkDatasetFormatter,
        'mustard': MUStARDDatasetFormatter,
        'annomi': AnnoMIDatasetFormatter
    }

    CONV_ID_COL_NAME = "conv_id"
    UTTER_ID_COL_NAME = "utter_id"
    UTTER_COL_NAME = "utterance"
    SPEAKER_ID_COL_NAME = "speaker"
    FILE_PATH_COL_NAME = "file_name"

    TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT = 0.8, 0.1, 0.1

    SAVED_DIR = "./data/BiMEmpDialogues"

    def __init__(self, meld_dataset_dir="./data/MELD/MELD",
                 dailytalk_datase_dir="./data/dailytalk/dailyTalk",
                 mustard_dataset_dir="./data/MUStARD/MUStARD",
                 annomi_dataset_dir="./data/AnnoMI/AnnoMI"):

        self.dataset_dirs = {
            'meld': meld_dataset_dir,
            'dailytalk': dailytalk_datase_dir,
            'mustard': mustard_dataset_dir,
            'annomi': annomi_dataset_dir
        }

    @classmethod
    def _change_column_names(cls, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        change columns names for having specified names for all datasets
        :param dataset_name:
        :param data:
        :return:
        """
        return data.rename(columns={
            cls.DATASETS[dataset_name].CONV_ID_COL_NAME: cls.CONV_ID_COL_NAME,
            cls.DATASETS[dataset_name].UTTER_ID_COL_NAME: cls.UTTER_ID_COL_NAME,
            cls.DATASETS[dataset_name].UTTER_COL_NAME: cls.UTTER_COL_NAME,
            cls.DATASETS[dataset_name].SPEAKER_ID_COL_NAME: cls.SPEAKER_ID_COL_NAME,
            cls.DATASETS[dataset_name].FILE_PATH_COL_NAME: cls.FILE_PATH_COL_NAME,
        })

    @classmethod
    def _change_conv_ids(cls, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        add a prefix for conv_ids based on
        :param dataset_name:
        :param data:
        :return:
        """
        conv_col_name = cls.CONV_ID_COL_NAME if cls.CONV_ID_COL_NAME in data.columns \
            else cls.DATASETS[dataset_name].CONV_ID_COL_NAME

        data[conv_col_name] = data[conv_col_name].apply(lambda x:
                                                        f"{list(cls.DATASETS.keys()).index(dataset_name)}_{str(x)}")
        return data

    @classmethod
    def _spilt_dataset(cls, data: pd.DataFrame) -> tuple:
        """
        split dataset to train, test, validation
        :param data:
        :return: train, val, test dataframes
        """
        conv_df = data[[cls.CONV_ID_COL_NAME]].groupby([cls.CONV_ID_COL_NAME]).count().reset_index()
        train_conv = conv_df.sample(frac=cls.TRAIN_SPLIT, random_state=200)
        test_val_conv = conv_df.drop(train_conv.index)
        test_conv = test_val_conv.sample(frac=cls.TEST_SPLIT/(1 - cls.TRAIN_SPLIT), random_state=200)
        val_conv = test_val_conv.drop(test_conv.index)

        train = pd.merge(data, train_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')
        test = pd.merge(data, test_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')
        val = pd.merge(data, val_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')

        return train, val, test

    def _file_struct_maker(self, data: pd.DataFrame, split='train') -> pd.DataFrame:
        """

        :param data:
        :param split:
        :return:
        """
        def move_file(path, conv_id, utter_idx) -> str:
            """
            create new path and move audio file to it
            :param path: path of audio
            :param conv_id: conversation id
            :param utter_idx: utterance id
            :return: new path
            """
            new_path = f"{self.SAVED_DIR}/{split}/audio/{conv_id}_{utter_idx}.wav"
            if not os.path.exists(f"{self.SAVED_DIR}/{split}/audio/"):
                os.makedirs(f"{self.SAVED_DIR}/{split}/audio/")
            if not os.path.exists(new_path):
                shutil.copy(path, new_path)
            return f"audio/{conv_id}_{utter_idx}.wav"

        data[self.FILE_PATH_COL_NAME] = data.apply(lambda x: move_file(path=x[self.FILE_PATH_COL_NAME],
                                                                       conv_id=x[self.CONV_ID_COL_NAME],
                                                                       utter_idx=x[self.UTTER_ID_COL_NAME]), axis=1)

        return data

    def run(self):
        """
        run merger management
        :return:
        """
        train_dfs, test_dfs, val_dfs = list(), list(), list()

        for dataset_name, path_dir in self.dataset_dirs.items():
            data = pd.read_csv(f"{path_dir}/metadata.csv")
            data = self._change_column_names(data=data, dataset_name=dataset_name)
            data = self._change_conv_ids(dataset_name=dataset_name, data=data)
            train, val, test = self._spilt_dataset(data=data)
            train_dfs.append(train)
            test_dfs.append(test)
            val_dfs.append(val)

        if not os.path.exists(self.SAVED_DIR):
            os.makedirs(self.SAVED_DIR)

        train_df = pd.concat(train_dfs)
        train_df = self._file_struct_maker(train_df, split='train')
        train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
        train_df.to_csv(f"{self.SAVED_DIR}/train/metadata.csv")

        test_df = pd.concat(test_dfs)
        test_df = self._file_struct_maker(test_df, split='test')
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
        test_df.to_csv(f"{self.SAVED_DIR}/test/metadata.csv")

        val_df = pd.concat(val_dfs)
        val_df = self._file_struct_maker(val_df, split='validation')
        val_df = val_df.loc[:, ~val_df.columns.str.contains('^Unnamed')]
        val_df.to_csv(f"{self.SAVED_DIR}/validation/metadata.csv")


if __name__ == '__main__':
    DatasetMerger().run()

    upload_to_hugging_face = False
    if upload_to_hugging_face:
        ds = load_dataset(DatasetMerger.SAVED_DIR)
        ds.push_to_hub(HUGGING_FACE_REPO_NAME, private=HUGGING_FACE_IS_PRIVATE, token=HUGGING_FACE_TOKEN)


