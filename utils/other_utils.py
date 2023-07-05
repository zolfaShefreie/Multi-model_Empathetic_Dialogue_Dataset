from zipfile import ZipFile
import pandas as pd
import os

from settings import PREFIX_MID_PROCESS_DIR, PREFIX_MID_PROCESS_CACHE_DIR


def unzip(zip_path: str, des_path: str):
    with ZipFile(zip_path, 'r') as z_obj:
        z_obj.extractall(path=des_path)
    return des_path


class WriterLoaderHandler:

    """
    this class allow to apply some changes into data by humans between automatic changes
    The supported format for data is pandas.DataFrame
    """
    PREFIX_PATH = PREFIX_MID_PROCESS_DIR
    CACHE_DIR = PREFIX_MID_PROCESS_CACHE_DIR
    SEP = "<SEP>"

    @classmethod
    def get_process_stage(cls, dataset_name: str, process_seq: list) -> int:
        """
        get the index of next process
        this functions show what is next stage for running
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :return:
        """
        dir_path = f"{cls.CACHE_DIR}/{dataset_name}"
        if os.path.exists(dir_path):
            # get all files in dir path
            files = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
            if files:
                # return the max of function_name_index
                # name files are like : dataset<SEP>func.csv
                return max([process_seq.index(file.strip('.csv')[0].strip(cls.SEP)[-1]) for file in files]) + 1
        return 0

    @classmethod
    def decorator(cls, dataset_name: str, process_seq: list, data_arg_name='data'):
        """
        create new decorator to handle write and load data before and after each functions
        WARNING: USE KEYWORDED ARGUMENTS iN RECALL FUNC
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :param data_arg_name: name of argument of data that used in function
        :return:new decoreator
        """

        def pre_pass_process(func):
            def new_func(*args, **kwargs):
                # get data form loading or arguments
                data = cls._get_entry_data(dataset_name=dataset_name, process_seq=process_seq,
                                           data_arg_name=data_arg_name, func_name=func.__name__, **kwargs)
                kwargs[data_arg_name] = data
                # run function
                new_data = func(*args, **kwargs)
                
                # save result in file
                cls._save_data(new_data, dataset_name, func_name=func.__name__)
                return new_data

            return new_func

        return pre_pass_process

    @classmethod
    def _save_data(cls, data: pd.DataFrame, dataset_name: str, func_name: str):
        """
        save data
        :param data: new data that function was processed
        :param dataset_name: name of dataset
        :param func_name: name of stage function
        :return:
        """
        data_path = cls._get_path(dataset_name=dataset_name, func_name=func_name)
        data.to_csv(data_path)
        cls._cache(data=data, dataset_name=dataset_name, func_name=func_name)
        cls._log(path=data_path, is_load_process=False)

    @classmethod
    def _cache(cls, data: pd.DataFrame, dataset_name: str, func_name: str):
        """
        save data in cache
        this folder is not allowed to be changed by user just used for automatic processes
        :param data: new data that function was processed
        :param dataset_name: name of dataset
        :param func_name: name of stage function
        :return:
        """
        data.to_csv(f"{cls.CACHE_DIR}/{dataset_name}/{dataset_name}{cls.SEP}{func_name}")

    @classmethod
    def _get_entry_data(cls, dataset_name: str, process_seq: list, func_name: str, data_arg_name: str,
                        **kwargs) -> pd.DataFrame:
        """
        load data or get data from arguments when it is the first stage
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :param func_name: name of stage function
        :param data_arg_name: name of argument of data that used in function
        :param kwargs: arguments of function
        :return:
        """
        # if it is not in first stage
        if func_name != process_seq[0]:
            
            # get path based on previous func_name
            previous_func_name = process_seq[process_seq.index(func_name) - 1]
            data_path = cls._get_path(dataset_name=dataset_name, func_name=previous_func_name)

            # is file is exists load data
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                cls._log(path=data_path, is_load_process=True)
                return data
        
        else:
            # if it is first stage or file doesn't exists
            return kwargs[data_arg_name]

    @classmethod
    def _get_path(cls, dataset_name: str, func_name: str) -> str:
        """
        get the path based on name of dataset and name of stage
        :param dataset_name: name of dataset
        :param func_name: name of stage function
        :return: path
        """
        return f"{cls.PREFIX_PATH}/{dataset_name}/{dataset_name}{cls.SEP}{func_name}.csv"

    @classmethod
    def _log(cls, path: str, is_load_process: bool):
        """
        with mini log for load and write
        :param path:
        :param is_load_process: a boolean that shows the process is save or load
        :return:
        """
        if is_load_process:
            print(f"data is read from {path}")
        else:
            print(f'data is written in {path}')
