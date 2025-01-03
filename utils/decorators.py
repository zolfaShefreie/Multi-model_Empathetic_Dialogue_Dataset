import pandas as pd
import os

from settings import PREFIX_MID_PROCESS_DIR, PREFIX_MID_PROCESS_CACHE_DIR


class WriterLoaderHandler:

    """
    this class is used for two requirements:
        1. caching between stages
        2. make possible to apply some changes into data by humans between automatic stages
    The supported format for data is pandas.DataFrame
    """
    PREFIX_PATH = PREFIX_MID_PROCESS_DIR
    CACHE_DIR = PREFIX_MID_PROCESS_CACHE_DIR
    SEP = "[SEP]"

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
                processed_stages = [file.split('.csv')[0].split(cls.SEP)[-1] for file in files
                                    if file.split('.csv')[0].split(cls.SEP)[-1] in process_seq]
                return max([process_seq.index(p_stage_name) for p_stage_name in processed_stages]) + 1
        return 0

    @classmethod
    def decorator(cls, dataset_name: str, process_seq: list, data_arg_name='data', human_editable: bool = False):
        """
        create new decorator to handle write and load data before and after each functions
        WARNING: USE KEYWORDED ARGUMENTS iN RECALL FUNC
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :param data_arg_name: name of argument of data that used in function
        :param human_editable: this is the function the humans can apply changes or not
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
                cls._save_data(data=new_data, dataset_name=dataset_name,
                               human_editable=human_editable,
                               func_name=func.__name__)
                return new_data

            new_func.__name__ = func.__name__
            return new_func

        return pre_pass_process

    @classmethod
    def _save_data(cls, data: pd.DataFrame, dataset_name: str, func_name: str, human_editable: bool = False):
        """
        save data
        :param data: new data that function was processed
        :param dataset_name: name of dataset
        :param func_name: name of stage function
        :param human_editable: this is the function the humans can apply changes or not
        :return:
        """
        if human_editable:
            data_path = cls.get_path(dataset_name=dataset_name, func_name=func_name, is_cache=False)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            columns = [col_name for col_name in data.columns if 'Unnamed' not in col_name]
            data[columns].to_csv(data_path, index=False)
            cls._log(path=data_path, is_load_process=False)
        cls._cache(data=data, dataset_name=dataset_name, func_name=func_name)

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
        data_path = cls.get_path(dataset_name=dataset_name, func_name=func_name, is_cache=True)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        columns = [col_name for col_name in data.columns if 'Unnamed' not in col_name]
        data[columns].to_csv(data_path, index=False)

    @classmethod
    def _get_entry_data(cls, dataset_name: str, process_seq: list, func_name: str, data_arg_name: str,
                        **kwargs) -> pd.DataFrame:
        """
        1. get data from arguments when it is the first stage or file doesn't exists
        2. load data from editable folder if doesn't exist from cache
        WARNING:
            The priorities are as follows:
                    1. load from editable folder
                    2. load from cache
                    3. get from arguments
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

            # if file is exists in editable folder, load data
            data_path = cls.get_path(dataset_name=dataset_name, func_name=previous_func_name, is_cache=False)
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                cls._log(path=data_path, is_load_process=True)
                return data

            else:
                # if file is exists in cache folder, load data
                data_path = cls.get_path(dataset_name=dataset_name, func_name=previous_func_name, is_cache=True)
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    return data

        # if it is first stage or file doesn't exists
        return kwargs[data_arg_name]

    @classmethod
    def get_path(cls, dataset_name: str, func_name: str, is_cache: bool = True) -> str:
        """
        get the path based on name of dataset and name of stage
        :param dataset_name: name of dataset
        :param func_name: name of stage function
        :param is_cache: a boolean that shows this address use for caching or not
        :return: path
        """
        prefix_dir = cls.CACHE_DIR if is_cache else cls.PREFIX_PATH
        return f"{prefix_dir}/{dataset_name}/{dataset_name}{cls.SEP}{func_name}.csv"

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

    @classmethod
    def add_decorator_to_func(cls, class_obj, dataset_name: str, process_seq: list, editable_process: list,
                              data_arg_name='data'):
        """
        add decorator to some functions of class
        :param class_obj: class or object of class
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :param data_arg_name: name of argument of data that used in function
        :param editable_process: list of process that edit by humans
        :return:
        """
        for function in process_seq:
            setattr(class_obj,
                    function,
                    WriterLoaderHandler.decorator(dataset_name=dataset_name,
                                                  process_seq=process_seq,
                                                  human_editable=True if function in editable_process else False,
                                                  data_arg_name=data_arg_name)(getattr(class_obj, function)))


class ChunkHandler:
    """
    this class control chunks based on columns for applying process
    this class using based on WriterLoaderHandler
    The supported format for data is pandas.DataFrame
    Warning:
        decorate before WriterLoaderHandler decoders
    """

    @classmethod
    def decorator(cls,
                  dataset_name: str,
                  group_by_keys: list,
                  chunk_len: int = None,
                  data_arg_name='data'):
        """
        create new decorator to handle slice and chunk data for apply process on it
        WARNING: USE KEYWORDS ARGUMENTS iN RECALL FUNC
        :param dataset_name: name of dataset
        :param group_by_keys: list of column names that we want slice data with same value in
        :param chunk_len: length of chunk
        :param data_arg_name: name of argument of data that used in function
        :return: decoreator function
        """
        def pre_pass_process(func):
            def new_func(*args, **kwargs):
                # get data form loading or arguments
                data = cls._prepare_chunk(dataset_name=dataset_name, group_by_keys=group_by_keys,
                                          chunk_length=chunk_len, data_arg_name=data_arg_name,
                                          func_name=func.__name__, **kwargs)
                kwargs[data_arg_name] = data
                # run function
                new_data = func(*args, **kwargs)

                # save result in file
                new_data = cls._prepare_result(data=new_data, dataset_name=dataset_name, func_name=func.__name__)
                return new_data

            new_func.__name__ = func.__name__
            return new_func

        return pre_pass_process

    @classmethod
    def _prepare_chunk(cls, dataset_name: str, group_by_keys: list, func_name: str, data_arg_name: str = 'data',
                       chunk_length: int = None, **kwargs) -> pd.DataFrame:
        """
        slice unprocessed data based on chunk length
        :param dataset_name: name of dataset
        :param group_by_keys: list of column names that we want slice data with same value in
        :param func_name: name of current function
        :param data_arg_name: name of argument of data that used in function
        :param chunk_length: length of chunk
        :param kwargs: arguments of function with their value
        :return: return a slice of data
        """
        data = kwargs[data_arg_name]
        if data is None or chunk_length is None:
            return data

        group_by_keys = data.columns if not group_by_keys else group_by_keys

        # get processed data to get unprocessed data
        data = cls._get_unprocessed_data(data=data, group_by_keys=group_by_keys, dataset_name=dataset_name,
                                         func_name=func_name)

        # chunk based on group_by_keys (data with same value in specific columns)
        group_data = data.groupby(group_by_keys).count().reset_index()[group_by_keys]
        group_data = group_data.iloc[:chunk_length, :]
        return pd.merge(data, group_data, on=group_by_keys, how='inner')

    @classmethod
    def _get_unprocessed_data(cls, data: pd.DataFrame, group_by_keys: list, dataset_name: str,
                              func_name: str) -> pd.DataFrame:
        """
        load processed data on exclude them from data
        :param data: entry data
        :param group_by_keys: list of column names that we want slice data with same value in
        :param dataset_name: name of dataset
        :param func_name: name of current function
        :return:
        """
        # get processed data to get unprocessed data
        file_path = WriterLoaderHandler.get_path(dataset_name=dataset_name, func_name=func_name, is_cache=True)
        if os.path.exists(file_path):
            processed_data = pd.read_csv(file_path)
            # get unprocessed data
            return pd.merge(data, processed_data, on=group_by_keys, how='left', indicator=True, suffixes=('', '_y')). \
                query("_merge == 'left_only'").reset_index(drop=True)[data.columns]
        return data

    @classmethod
    def unprocessed_record_number(cls, dataset_name: str, func_name: str, pre_func_name: str) -> tuple:
        """
        count the unprocessed record
        :param dataset_name: name of dataset
        :param func_name: name of current function
        :param pre_func_name: name of previous function
        :return: return number of unprocessed record if there is no data in previous stage return -1
        """
        pre_stage_file_name = WriterLoaderHandler.get_path(dataset_name=dataset_name, func_name=pre_func_name,
                                                           is_cache=True)
        if os.path.exists(pre_stage_file_name):
            data = pd.read_csv(pre_stage_file_name)
            return len(cls._get_unprocessed_data(data=data, dataset_name=dataset_name, func_name=func_name,
                                                 group_by_keys=list(data.columns))), len(data)

        else:
            return -1, -1

    @classmethod
    def _prepare_result(cls, data: pd.DataFrame, dataset_name: str, func_name: str) -> pd.DataFrame:
        """
        merge processed data with previous data on file
        :param data: new data that function was processed
        :param dataset_name: name of dataset
        :param func_name: name of current function
        :return: merged data
        """
        file_path = WriterLoaderHandler.get_path(dataset_name=dataset_name, func_name=func_name, is_cache=False)
        if not os.path.exists(file_path):
            file_path = WriterLoaderHandler.get_path(dataset_name=dataset_name, func_name=func_name, is_cache=True)
            if not os.path.exists(file_path):
                return data

        old_data = pd.read_csv(file_path)
        return pd.concat([old_data, data], ignore_index=True, sort=False)

    @classmethod
    def add_decorator_to_func(cls, class_obj,
                              dataset_name: str,
                              process_seq: list,
                              group_by_keys: list,
                              chunk_length: int = None,
                              data_arg_name='data'):
        """
        add decorator to some functions of class
        :param class_obj: class or object of class
        :param dataset_name: name of dataset
        :param process_seq: list of process that sorted by running turn
        :param group_by_keys: list of column names that we want slice data with same value in
        :param data_arg_name: name of argument of data that used in function
        :param chunk_length: length of chunk
        :return:
        """
        for function in process_seq:
            setattr(class_obj,
                    function,
                    ChunkHandler.decorator(dataset_name=dataset_name,
                                           data_arg_name=data_arg_name,
                                           chunk_len=chunk_length,
                                           group_by_keys=group_by_keys)(getattr(class_obj, function)))

