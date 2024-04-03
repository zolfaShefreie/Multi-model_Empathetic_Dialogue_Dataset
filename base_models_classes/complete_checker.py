import enum
import re
from openai import OpenAI

from utils.llms_utils import LLMsCompletionService
from settings import FARAROOM_AUTH_CONFIG, OPENAI_API_KEY, OPENAI_MODEL, TOGETHER_API_KEY, TOGETHER_MODEL


class CompleteCheckerClassifierLLMs:
    """this class uses LLMs and prompt engineering for check the conversation if it is complete or not"""

    class CompleteEnum(enum.Enum):
        complete = 1
        incomplete = 0

    SYS_PROMPT = "You are an AI expert in human communications such as conversation. " \
                 "You are tasked to check carefully that a conversation is complete or incomplete."

    USER_QUERY = "i give you a conversation and you will tell me it is complete or incomplete conversation based on " \
                 "its context and explain reason behind it. consider a complete conversation is a conversation in " \
                 "which there is no information gap, and there is no room for ambiguity, and the necessary " \
                 "information to understand the topic is explicitly or implicitly provided, with logical coherence" \
                 " between the utterances of both parties. you have to " \
                 "answer following this template: Reason: [reason] \nThe final answer is: [complete or incomplete]" \
                 "\nconversation:\n{conversation}\n" \
                 "Answer: let's think step by step to reach the right conclusion."

    REGEX = r"(Reason|\[Reason\]|reason): (.+)( |\n)*((T|t)he final answer is):? (complete|incomplete|Complete|Incomplete)"

    REASON_KEY_NAME = "Reason"
    LABEL_KEY_NAME = "Label"
    NUMBER_RESULT = 40
    NUMBER_LOOP_WITHOUT_RESULT = 6
    REQUEST_SLEEP = 1000
    COMPLETE_RATE = 0.4

    @classmethod
    def get_conversation_prompt(cls, conv_str, chat_form=False):
        """
        get the final prompt based on task (text or chat)
        :param conv_str: str of conv
        :param chat_form: is task chat completion_?
        :return: str or a list
        """
        if chat_form:
            return [
                {"role": "system", "content": cls.SYS_PROMPT},
                {"role": "user", "content": cls.USER_QUERY.format(conversation=conv_str)}
            ]

        else:
            return f"{cls.SYS_PROMPT}\n{cls.USER_QUERY.format(conversation=conv_str)}"

    @classmethod
    def extract_llms_response_info(cls, response: str) -> tuple:
        """
        extract label and reason from a response
        :param response: response of LLMs
        :return: label, reason
        """
        match_result = re.search(cls.REGEX, response.replace("\n", " "))
        if match_result is not None:
            # based on regex the 6th group is label and 2nd group is reason
            return match_result.group(6), match_result.group(2)
        return None, None

    @classmethod
    def get_response(cls, conv_str: str, tool: enum.Enum = LLMsCompletionService.Tools.FARAROOM,
                     num_requests: int = NUMBER_RESULT) -> list:
        """
        get response of LLMs
        :param num_requests: number of requests for each conversation
        :param conv_str: str of conv
        :param tool: which tool do you want to use for completion task?
        :return:
        """
        if tool != LLMsCompletionService.Tools.OPENAI:
            model = None if tool == LLMsCompletionService.Tools.FARAROOM else TOGETHER_MODEL
            tool_auth_info = FARAROOM_AUTH_CONFIG if tool == LLMsCompletionService.Tools.FARAROOM else TOGETHER_API_KEY
            prompt = cls.get_conversation_prompt(conv_str=conv_str, chat_form=False)

            return LLMsCompletionService.completion(tool_auth_info=tool_auth_info,
                                                    prompt=prompt,
                                                    model=model,
                                                    number_of_choices=num_requests,
                                                    request_sleep=cls.REQUEST_SLEEP,
                                                    completion_kind=LLMsCompletionService.CompletionKinds.TEXT,
                                                    tool=tool)
        else:
            messages = cls.get_conversation_prompt(conv_str=conv_str, chat_form=True)
            return LLMsCompletionService.completion(tool_auth_info=OpenAI(api_key=OPENAI_API_KEY),
                                                    messages=messages,
                                                    model=OPENAI_MODEL,
                                                    number_of_choices=num_requests,
                                                    completion_kind=LLMsCompletionService.CompletionKinds.CHAT,
                                                    tool=tool)

    @classmethod
    def aggregate_responses(cls,
                            all_req_labels: list,
                            all_req_reasons: list,
                            complete_key_name: str = 'is_completed',
                            reasons_key_name: str = 'complete_check_reasons',
                            percent_key_name: str = 'complete_check_percent') -> dict:
        """
        aggregate responses and get label, reasons and percent of each label for each conversation
        :param all_req_labels: list of label
        :param all_req_reasons: list of reason
        :param percent_key_name: the key name of percents. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param complete_key_name:the key name of complete_check_label. use for result
        :return: label
        """

        def key_of_label(extracted_label: str) -> str:
            """
            get the key based on extracted label
            :param extracted_label: extracted label
            :return: name of the key
            """
            if extracted_label.lower() == cls.CompleteEnum.incomplete.name.lower():
                return cls.CompleteEnum.incomplete.name.lower()
            if extracted_label.lower() == cls.CompleteEnum.complete.name.lower():
                return cls.CompleteEnum.complete.name.lower()

            return ''

        def get_max_label_reason(data: dict,
                                 complete_key_name: str = 'is_completed',
                                 reasons_key_name: str = 'complete_check_reasons',
                                 percent_key_name: str = 'complete_check_percent') -> dict:
            """
            find the maximum label and return with its reason and all percents
            :param percent_key_name: the key name of percents. use for result
            :param reasons_key_name: the key name of reasons. use for result
            :param complete_key_name:the key name of complete_check_label. use for result
            :param data: info dictionary
            :return: max_value, it's reason and percents
            """
            sum_request = sum([label_info['number'] for label_info in data.values()])
            avg_label = {cls.CompleteEnum[key].name: 0 if sum_request == 0 else label_info['number']/sum_request
                         for key, label_info in data.items()}
            max_key_number = max(avg_label, key=avg_label.get)
            return {complete_key_name: cls.CompleteEnum[max_key_number].value,
                    reasons_key_name: data[max_key_number]['reasons'],
                    percent_key_name: avg_label}

        # init of info
        complete_info = {cls.CompleteEnum.complete.name: {'number': 0, 'reasons': list()},
                         cls.CompleteEnum.incomplete.name: {'number': 0, 'reasons': list()}}

        # update data for each response
        for label, reason in zip(all_req_labels, all_req_reasons):
            complete_label = key_of_label(label)
            complete_info[complete_label]['number'] = complete_info[complete_label]['number'] + 1
            complete_info[complete_label]['reasons'] = complete_info[complete_label]['reasons'] + [reason]

        return get_max_label_reason(data=complete_info, complete_key_name=complete_key_name,
                                    reasons_key_name=reasons_key_name, percent_key_name=percent_key_name)

    @classmethod
    def continue_send_requests(cls, complete_count: int, number_request: int) -> bool:
        """
        is ot enough completed response or request again?
        if the some response are incomplete but the number of complete request is more than specific number,
        it will be enough else it won't be
        :param complete_count: number of complete request is done in right condition
        :param number_request: number of requests for each conversation
        :return: ture or false (continue => true => run the loop)
        """
        complete_fix_number = max(1, int(number_request * cls.COMPLETE_RATE))
        return True if complete_count < complete_fix_number else False

    @classmethod
    def run_process(cls,
                    conv_str: str,
                    number_request: int = NUMBER_RESULT,
                    tool: enum.Enum = LLMsCompletionService.Tools.FARAROOM,
                    complete_key_name: str = 'is_completed',
                    reasons_key_name: str = 'complete_check_reasons',
                    percent_key_name: str = 'complete_check_percent'):
        """
        this function plays as management of whole process:
            1. send request to get llms responses for conv_str
            2. extracted info from each response
            3. check if the response has the data based on prompt template
            4. aggregate the result
            5. return all result

        :param conv_str: str of conversation
        :param number_request: number of requests for each conversation
        :param tool: which tool do you want to use for completion task?
        :param complete_key_name: the key name of complete_check_label. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param percent_key_name: the key name of percents. use for result
        :return:
        """
        all_labels, all_reasons = list(), list()
        incomplete_count = 0
        loop_without_result = 0
        while cls.continue_send_requests(complete_count=len(all_labels), number_request=number_request):
            responses = cls.get_response(conv_str=conv_str, tool=tool,
                                         num_requests=number_request if incomplete_count == 0 else incomplete_count)

            # save incomplete count at previous loop. if there is no pre_loop set -1
            pre_incomplete_count = incomplete_count if incomplete_count != 0 else -1
            # reset incomplete_count for the current running loop
            incomplete_count = 0
            for response in responses:
                label, reason = cls.extract_llms_response_info(response.text)
                if label is not None:
                    all_labels.append(label)
                    all_reasons.append(reason)
                else:
                    incomplete_count += 1
            print('incomplete responses', incomplete_count)

            # check if this loop has no new result. add to a counter
            # else reset the counter
            # if the counter gets max number, loop will stop
            if pre_incomplete_count == incomplete_count:
                loop_without_result += 1
            else:
                loop_without_result = 0

            if loop_without_result >= cls.NUMBER_LOOP_WITHOUT_RESULT:
                print("break the loop")
                break

        return cls.aggregate_responses(all_req_labels=all_labels, all_req_reasons=all_reasons,
                                       complete_key_name=complete_key_name,
                                       percent_key_name=percent_key_name, reasons_key_name=reasons_key_name)
