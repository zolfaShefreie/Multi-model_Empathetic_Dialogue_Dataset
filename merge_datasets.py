import os

import pandas as pd

from dataset_format_converter.dataset_process import MELDDatasetFormatter, DailyTalkDatasetFormatter, \
    MUStARDDatasetFormatter, AnnoMIDatasetFormatter


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
    FILE_PATH_COL_NAME = "file-path"

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
        print(conv_df.columns)
        train_conv = conv_df.sample(frac=cls.TRAIN_SPLIT, random_state=200)
        test_val_conv = conv_df.drop(train_conv.index)
        test_conv = test_val_conv.sample(frac=cls.TEST_SPLIT, random_state=200)
        val_conv = test_val_conv.drop(test_conv.index)

        train = pd.merge(data, train_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')
        test = pd.merge(data, test_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')
        val = pd.merge(data, val_conv[[cls.CONV_ID_COL_NAME]], on=[cls.CONV_ID_COL_NAME], how='inner')

        return train, val, test

    def run(self):
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
        train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
        train_df.to_csv(f"{self.SAVED_DIR}/train.csv")

        test_df = pd.concat(test_dfs)
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
        test_df.to_csv(f"{self.SAVED_DIR}/test.csv")

        val_df = pd.concat(val_dfs)
        val_df = val_df.loc[:, ~val_df.columns.str.contains('^Unnamed')]
        val_df.to_csv(f"{self.SAVED_DIR}/validation.csv")


if __name__ == '__main__':
    DatasetMerger().run()


