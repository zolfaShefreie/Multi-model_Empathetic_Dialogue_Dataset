import enum
import json
import requests
import together
import openai
import time


class LLMsCompletionService:

    class Response:
        def __init__(self, text):
            self.text = text
            self._correct_some_char()

        def _correct_some_char(self):
            self.text = self.text.replace('\\n', '\n')
            self.text = self.text.replace('\\t', '\t')

    class Tools(enum.Enum):
        TOGETHER = 'together'
        OPENAI = 'openai'
        FARAROOM = 'fararoom'

    class CompletionKinds(enum.Enum):
        TEXT = 'text'
        CHAT = 'chat'

    DEFAULT_CONFIG = {
        'temperature': 0.7,
        # 'top_p': 0.95,
        'max_tokens': 512,
    }

    ERROR_SLEEP = 1000

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
        :param request_sleep: time of sleep in milliseconds between two requests
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
                  'number_of_choices': number_of_choices,
                  'request_sleep': request_sleep}
        try:
            return getattr(cls, f"_completion_{completion_kind.value}_{tool.value}")(**kwargs)
        except Exception as e:
            print('request failed', e)
            time.sleep(cls.ERROR_SLEEP * 0.001)
            return cls.completion(tool_auth_info=tool_auth_info, model=model, prompt=prompt, messages=messages,
                                  tool=tool, completion_kind=completion_kind, number_of_choices=number_of_choices,
                                  config=config, request_sleep=request_sleep)

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
                                  number_of_choices: int = 1,
                                  request_sleep: int = 100,
                                  **kwargs) -> list:
        """
        get response of model using together.ai
        :param tool_auth_info: API_KEY of together.ai account
        :param model: name of model
        :param prompt: for text completion task
        :param config: a dict for setting some configs of completion task like top_k
        :param number_of_choices: number of return result for completion task
        :param request_sleep: time of sleep in milliseconds between two requests
        :return: a list of responses
        """
        together.api_key = tool_auth_info
        responses = list()

        for i in range(number_of_choices):

            response = together.Complete.create(model=model,
                                                prompt=prompt,
                                                **config)
            responses.append(response)
            time.sleep(request_sleep * 0.001)

        return cls._response_reformatting(responses=responses, tool=cls.Tools.TOGETHER,
                                          task=cls.CompletionKinds.TEXT)

    @classmethod
    def _completion_text_fararoom(cls,
                                  tool_auth_info: dict,
                                  prompt: str,
                                  number_of_choices: int = 1,
                                  request_sleep: int = 100,
                                  **kwargs) -> list:
        """
        get the response of gpt-3.5 using fararoom tool
        :param tool_auth_info: a dictionary of auth info
        :param prompt: for text completion task
        :param number_of_choices: number of return result for completion task
        :param request_sleep: time of sleep in milliseconds between two requests
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

        responses = list()

        for i in range(number_of_choices):
            response = requests.request("POST", url, headers=headers, data=payload)
            responses.append(response)
            time.sleep(request_sleep * 0.001)

        return cls._response_reformatting(responses=responses, tool=cls.Tools.FARAROOM,
                                          task=cls.CompletionKinds.TEXT)

    @classmethod
    def _completion_text_openai(cls,
                                tool_auth_info,
                                prompt: str,
                                model: str,
                                config: dict,
                                number_of_choices: int = 1,
                                **kwargs) -> list:
        """
        get the response of model using openai tool and text completion task
        :param tool_auth_info: client of openai
        :param prompt: for text completion task
        :param model: name of model
        :param config: a dict for setting some configs of completion task like top_k
        :param number_of_choices: number of return result for completion task
        :return: a list of responses
        """

        response = tool_auth_info.completions.create(model=model,
                                                     prompt=prompt,
                                                     n=number_of_choices,
                                                     **config)
        return cls._response_reformatting(responses=response, tool=cls.Tools.OPENAI,
                                          task=cls.CompletionKinds.TEXT)

    @classmethod
    def _completion_chat_openai(cls,
                                tool_auth_info,
                                messages: list,
                                model: str,
                                config: dict,
                                number_of_choices: int = 1,
                                **kwargs) -> list:
        """
        get the model response using openai tool and chat completion task
        :param tool_auth_info: client of openai
        :param messages: for completion chat task
        :param model: name of model
        :param config: a dict for setting some configs of completion task like top_k
        :param number_of_choices: number of return result for completion task
        :return: a list of responses
        """

        response = tool_auth_info.chat.completions.create(model=model,
                                                          messages=messages,
                                                          n=number_of_choices,
                                                          **config)
        return cls._response_reformatting(responses=response, tool=cls.Tools.OPENAI,
                                          task=cls.CompletionKinds.CHAT)

    @classmethod
    def _response_reformatting(cls,
                               responses,
                               tool: enum.Enum,
                               task: enum.Enum = CompletionKinds.TEXT) -> list:
        """
        get te responses of tools and convert it to new format
        :param responses: the responses of LLMs
        :param tool: which tool did you used? choices = Tools enum class
        :param task: which task did you used? choices = CompletionKinds enum class
        :return: a list of response
        """
        new_format_responses = list()

        if tool == cls.Tools.FARAROOM:
            for result in responses:
                new_format_responses.append(cls.Response(text=result.text))
            return new_format_responses

        elif tool == cls.Tools.OPENAI:

            if task == cls.CompletionKinds.TEXT:
                for result in responses.choices:
                    new_format_responses.append(cls.Response(text=result.text))
                return new_format_responses

            else:
                for result in responses.choices:
                    new_format_responses.append(cls.Response(text=result.message.content))
                return new_format_responses

        else:
            # together
            for result in responses:
                new_format_responses.append(cls.Response(text=result['output']['choices'][0]['text']))
            return new_format_responses