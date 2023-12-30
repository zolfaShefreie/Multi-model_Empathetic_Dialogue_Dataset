import enum
import json
from zipfile import ZipFile
import pandas as pd
import os
import requests
import together
import openai

from settings import PREFIX_MID_PROCESS_DIR, PREFIX_MID_PROCESS_CACHE_DIR


def unzip(zip_path: str, des_path: str):
    with ZipFile(zip_path, 'r') as z_obj:
        z_obj.extractall(path=des_path)
    return des_path


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
                return max([process_seq.index(file.split('.csv')[0].split(cls.SEP)[-1]) for file in files]) + 1
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
            data.to_csv(data_path)
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
        data.to_csv(data_path)

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


class LLMsCompletionService:

    class Tools(enum.Enum):
        TOGETHER = 'together'
        OPENAI = 'openai'
        FARAROOM = 'fararoom'

    class CompletionKinds(enum.Enum):
        TEXT = 'text'
        CHAT = 'chat'

    DEFAULT_CONFIG = {
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 512,
    }

    @classmethod
    def completion(cls,
                   tool_auth_info,
                   model: str = None,
                   prompt: str = None,
                   messages: list = None,
                   tool: enum.Enum = Tools.FARAROOM,
                   completion_kind: enum.Enum = CompletionKinds.TEXT,
                   number_of_choices: int = 1,
                   config: dict = None,
                   request_sleep: int = 100) -> list:
        """
        interface for using tools for completion task
        :param tool_auth_info: API_KEY or dict of info or a client for using one of tools that you are using for this request
        :param request_sleep: time of sleep between two requests
        :param model: name of model
        :param prompt: for text completion task
        :param messages: for completion chat task
        :param tool: which tool do you want to use for completion task?
        :param completion_kind: which kind of completion do you want to use?
        :param number_of_choices: number of return result for completion task
        :param config: a dict for setting some configs of completion task like top_k
        :return: a list of response
        """

        cls._validate_entry(model=model, prompt=prompt, messages=messages, tool=tool, completion_kind=completion_kind)

        config = config if config is not None else cls.DEFAULT_CONFIG
        kwargs = {'config': config if config is not None else cls.DEFAULT_CONFIG,
                  'tool_auth_info': tool_auth_info,
                  'model': model,
                  'prompt': prompt,
                  'messages': messages,
                  'number_of_choices': number_of_choices}

        return getattr(cls, f"_completion_{completion_kind.value}_{tool.value}")(**kwargs)

    @classmethod
    def _validate_entry(cls,
                        model: str = None,
                        prompt: str = None,
                        messages: list = None,
                        tool: enum.Enum = Tools.FARAROOM,
                        completion_kind: enum.Enum = CompletionKinds.TEXT):
        """
        validate the entry of cls.completion function
        :param model: name of model
        :param prompt: for text completion task
        :param messages: for completion chat task
        :param tool: which tool do you want to use for completion task?
        :param completion_kind: which kind of completion do you want to use?
        :return:
        """
        # check prompt and messages to make sure they aren't empty based on task
        if (prompt is None or len(prompt) == 0) and completion_kind == cls.CompletionKinds.TEXT:
            raise Exception("prompt is Empty and the task is text completion so you must enter prompt as an input")
        if (messages is None or len(messages) == 0) and completion_kind == cls.CompletionKinds.CHAT:
            raise Exception("messages is Empty and the task is chat completion so you must enter messages as an input")

        # check model_name isn't empty for openai and together
        if (model is None or len(model) == 0) and tool != cls.Tools.FARAROOM:
            raise Exception(f"you must specify model for {cls.Tools.OPENAI.value} and {cls.Tools.TOGETHER.value}")

        # check task and tool
        if getattr(cls, f"_completion_{completion_kind.value}_{tool.value}", None) is None:
            raise Exception(f"{tool.name} doesn't support {completion_kind.name} completion")

    @classmethod
    def _completion_text_together(cls,
                                  tool_auth_info: str,
                                  model: str,
                                  prompt: str,
                                  config: dict,
                                  **kwargs) -> list:
        """

        :param tool_auth_info: API_KEY of together.ai account
        :param model: name of model
        :param prompt: for text completion task
        :param config: a dict for setting some configs of completion task like top_k
        :return: a list of responses
        """
        together.api_key = tool_auth_info

        response = together.Complete.create(model=model,
                                            prompt=prompt,
                                            **config)
        return response['output']['choices']

    @classmethod
    def _completion_text_fararoom(cls,
                                  tool_auth_info: dict,
                                  prompt: str,
                                  **kwargs) -> list:
        """

        :param tool_auth_info: a dictionary of auth info
        :param prompt: for text completion task
        :return: a list of responses
        """

        url = "https://api.fararoom.ir/dotask/"

        payload = json.dumps({
            "task_name": tool_auth_info['FARAROOM_TASK_NAME'],
            "source": tool_auth_info['FARAROOM_SOURCE'],
            "meta": tool_auth_info['FARAROOM_META'],
            "content": prompt,
        })

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {tool_auth_info["FARAROOM_TOKEN"]}'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return [response.text]

    @classmethod
    def _completion_text_openai(cls,
                                tool_auth_info,
                                prompt: str,
                                model: str,
                                config: dict,
                                **kwargs) -> list:
        """

        :param tool_auth_info: client of openai
        :param prompt: for text completion task
        :param model: name of model
        :param config: a dict for setting some configs of completion task like top_k
        :return: a list of responses
        """

        response = tool_auth_info.completions.create(model=model,
                                                     prompt=prompt,
                                                     **config)
        return response.choices

    @classmethod
    def _completion_chat_openai(cls,
                                tool_auth_info,
                                messages: list,
                                model: str,
                                config: dict,
                                **kwargs) -> list:
        """

        :param tool_auth_info: client of openai
        :param messages: for completion chat task
        :param model: name of model
        :param config: a dict for setting some configs of completion task like top_k
        :return: a list of responses
        """

        response = tool_auth_info.chat.completions.create(model=model,
                                                          messages=messages,
                                                          **config)
        return response.choices
