from transformers import RobertaModel, RobertaTokenizer
import torch
import lightning.pytorch as pl
import torchmetrics
import enum
import re
from openai import OpenAI

from utils import model_utils
from utils import util_transforms
from utils.decorators import LLMsCompletionService
from settings import EMPATHY_KIND_MODEL_FILE_PATH, FARAROOM_AUTH_CONFIG, OPENAI_API_KEY, OPENAI_MODEL, \
    TOGETHER_API_KEY, TOGETHER_MODEL


class EmpathyKindEnum(enum.Enum):
    NONE = 0
    SEEKING = 1
    PROVIDING = 2


class EmpathyKindRobertaModel(pl.LightningModule):
    num_classes = 3
    LOSS = torch.nn.BCEWithLogitsLoss()

    def __init__(self):
        super().__init__()
        self.transformer_model = RobertaModel.from_pretrained("roberta-base")
        self.drop = torch.nn.Dropout(0.4)
        self.out = torch.nn.Linear(768, self.num_classes)

    def forward(self, ids, mask, token_type_ids):
        x = self.transformer_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[0].mean(dim=1)
        x = self.drop(x)
        output = self.out(x)

        return output

    def training_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_loss": loss, "train_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        val_loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        self.log_dict({"val_loss": val_loss, "val_accuracy": acc}, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        ids, mask, token_type_ids, y = batch
        pred = self(ids, mask, token_type_ids)
        test_loss = self.LOSS(pred, y.float())
        acc = torchmetrics.functional.classification.multilabel_accuracy(pred, y.float(), num_labels=3)
        f1_score = torchmetrics.functional.classification.multilabel_f1_score(pred, y.float(), num_labels=3)
        self.log_dict({"test_loss": test_loss, "test_accuracy": acc, "test_f1": f1_score},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return test_loss

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0):
        return self(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class EmpathyKindClassifier(model_utils.BaseDeployedModel):

    def _get_checkpoint_path(self) -> str:
        """
        :return: path of model checkpoint
        """
        return EMPATHY_KIND_MODEL_FILE_PATH

    def _get_model_class(self):
        """
        :return: model class
        """
        return EmpathyKindRobertaModel

    def _get_data_pre_process_list(self) -> list:
        """"
        :return: a list of process to apply to input data
        """
        return [
            util_transforms.TextCleaner(have_label=False),
            util_transforms.Tokenizer(tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                                      have_label=False),
            util_transforms.ToTensor(),
        ]

    def _get_result_after_process_list(self) -> list:
        """
        :return: a list of process to apply to model output
        """
        return [
            torch.nn.functional.sigmoid,
            util_transforms.IntegerConverterWithIndex(dim=1)
        ]

    def get_arg_index_model_input(self):
        """
        :return: kwargs format of model()
        """
        return {
            'ids': 0,
            'mask': 1,
            'token_type_ids': 2
        }


class EmpathyKindClassifierLLMs:
    """this class uses LLMs and prompt engineering for classify utterances with EmpathyKind labels"""

    SYS_PROMPT = "You are an AI expert in human communications such as conversation and you can understand empathy in" \
                 "conversations.\nYou are tasked analyzing conversation carefully and annotate each utterance based " \
                 "on its previous utterances as the content with three labels" \
                 ' "empathy_seeker", "empathy_provider" and "none" and explain the reasoning behind each annotataion.'

    USER_QUERY = "Empathetic conversation involves actively listening, understanding, and responding with care to " \
                 "someone's shared thoughts, feelings, and experiences without judgment. We have three empathy " \
                 "communication mechanisms: emotional reactions, interpretations, and explorations, each with its " \
                 "specific intents. Emotional reactions involve expressing warmth, compassion, and concern, and " \
                 "include intents such as Consoling, Sympathizing, Wishing, Expressing relief, Expressing care or" \
                 " concern, Appreciating, Encouraging. Interpretations focus on understanding the conversation " \
                 "partner's feelings and thoughts, with intents like Agreeing, Acknowledging, Sharing own " \
                 "thoughts/opinion, Sharing or relating to own experience, Suggesting, Advising, Disapproving. " \
                 "Explorations entail asking questions to deepen understanding and display interest, " \
                 "aiming to elicit more information and engagement. By utilizing these mechanisms and intents, " \
                 "individuals can create meaningful connections and support others effectively in their " \
                 "conversations. Consider Task-oriented conversation focuses on achieving goals so it is not " \
                 "empathetic conversation. I will give you a conversation and you will analyze it, step-by-step, " \
                 "to classify each utterance based on its previous utterances as the content with three labels " \
                 '"empathy_seeker", "empathy_provider" and "none". If speaker describes and shares personal ' \
                 'experiences, feelings and thoughts, label utterance as "empathy_seeker", if speaker shows' \
                 " the understanding of others' feelings, thoughts, or attitudes using explained empathy " \
                 'communication mechanisms, label utterance as "empathy_provider" and Utterances that ' \
                 "don't fall into these categories are labeled as " \
                 '"none" For the given conversation, classify each utterance with three label "empathy_seeker", ' \
                 '"empathy_provider" and "none" and explain the reasoning behind each classification and You will ' \
                 'answer following this template:\n[index]: \nReason: [reason] \nLabel: [label] :\n' \
                 "{conversation}\nAnswer: Let's think step by step to reach the right conclusion"

    REGEX = r"(index:)?(\d+|\[\d+\])(\.|:)( |\n)?.*( |\n)?(((Reason|\[Reason\]): (.+)( |\n)?(Label|\[Label\]): " \
            r"(None|none|Empathy_seeker|empathy_provider|Empathy_provider|empathy_seeker)\.?)|((Label|\[Label\]): " \
            r"(None|none|Empathy_seeker|empathy_provider|Empathy_provider|empathy_seeker)\.?( |\n)?(Reason|\[Reason\]" \
            r"): (.+)))"

    REASON_KEY_NAME = "Reason"
    LABEL_KEY_NAME = "Label"
    NUMBER_RESULT = 40
    REQUEST_SLEEP = 1000

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
        extract label and reason for each utterance
        :param response: response of LLMs
        :return: label list, reason list
        """
        labels, reasons = list(), list()

        match_result = re.findall(cls.REGEX, response)

        for each_match in match_result:
            # based on regex each_match[6] and each_match[12] can contain string of reason + label
            # each_match[x] means x'th group of regex
            # if reason + label are in each_match[6] then each_match[12] is empty
            # and if label + reason  are in each_match[12] then each_match[6] is empty
            # in each_mach[6] => reason + label and reason is each_match[8] and label is each_match[11]
            # in each_mach[12] => label + reason and label is each_match[14] and reason is each_match[17]
            if each_match[6] or each_match[12]:
                reasons.append(each_match[8] if each_match[6] else each_match[17])
                labels.append(each_match[11] if each_match[6] else each_match[14])

        return labels, reasons

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
                                                    number_of_choices=cls.NUMBER_RESULT,
                                                    request_sleep=num_requests,
                                                    completion_kind=LLMsCompletionService.CompletionKinds.TEXT,
                                                    tool=tool)
        else:
            messages = cls.get_conversation_prompt(conv_str=conv_str, chat_form=True)
            return LLMsCompletionService.completion(tool_auth_info=OpenAI(api_key=OPENAI_API_KEY),
                                                    messages=messages,
                                                    model=OPENAI_MODEL,
                                                    number_of_choices=cls.NUMBER_RESULT,
                                                    completion_kind=LLMsCompletionService.CompletionKinds.CHAT,
                                                    tool=tool)

    @classmethod
    def aggregate_responses(cls,
                            all_req_labels: list,
                            all_req_reasons: list,
                            number_of_utter: int,
                            empathy_key_name: str = 'Empathy',
                            reasons_key_name: str = 'empathy_reasons',
                            percent_key_name: str = 'empathy_percents') -> list:
        """
        aggregate responses and get label, reasons and percent of each label for each conversation
        :param number_of_utter: number of utterances in conversation
        :param all_req_labels: list of labels(contains label of each utterance)
        :param all_req_reasons: list of reasons(contains reason of label assignment of each utterance)
        :param percent_key_name: the key name of percents. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param empathy_key_name: the key name of empathy_label. use for result
        :return: label
        """
        def key_of_label(label: str):
            """
            get the key based on extracted label
            :param label: extracted label
            :return: name of the key
            """
            for key in [EmpathyKindEnum.SEEKING.name, EmpathyKindEnum.PROVIDING.name, EmpathyKindEnum.NONE.name]:
                if key.lower() in label.lower():
                    return key

        def get_max_label_reason(data: dict,
                                 empathy_key_name: str = 'Empathy',
                                 reasons_key_name: str = 'empathy_reasons',
                                 percent_key_name: str = 'empathy_percents') -> dict:
            """
            find the maximum label and return with its reason and all percents
            :param percent_key_name: the key name of percents. use for result
            :param reasons_key_name: the key name of reasons. use for result
            :param empathy_key_name: the key name of empathy_label. use for result
            :param data: info for each item of label_utter_info => data of each utter
            :return: max_value, it's reason and percents
            """
            avg_label = {EmpathyKindEnum[empathy_key].value: label_info['number']/len(data.keys())
                         for empathy_key, label_info in data.items()}
            max_key_number = max(avg_label, key=avg_label.get)
            return {empathy_key_name: EmpathyKindEnum[max_key_number].value,
                    reasons_key_name: data[max_key_number]['reasons'],
                    percent_key_name: avg_label}

        # init of info for each utterance
        label_utter_info = {i: {EmpathyKindEnum.SEEKING.name: {'number': 0, 'reasons': list()},
                                EmpathyKindEnum.PROVIDING.name: {'number': 0, 'reasons': list()},
                                EmpathyKindEnum.NONE.name: {'number': 0, 'reasons': list()}}
                            for i in range(number_of_utter)}

        # update data for each response and each utterance
        for index, labels, reasons in enumerate(zip(all_req_labels, all_req_reasons)):
            for extracted_label, reason in zip(labels, reasons):
                empathy_kind = key_of_label(extracted_label)
                label_utter_info[index][empathy_kind]['number'] = label_utter_info[index][empathy_kind]['number'] + 1
                label_utter_info[index][empathy_kind]['reason'] = label_utter_info[index][empathy_kind]['reason'] + \
                                                                  [reason]

        return [get_max_label_reason(data=utter, empathy_key_name=empathy_key_name,
                                     reasons_key_name=reasons_key_name, percent_key_name=percent_key_name)
                for utter in label_utter_info]

    @classmethod
    def is_complete(cls, labels: list, number_of_utter: int) -> bool:
        """
        :param labels: extracted labels of a request)
        :param number_of_utter: number of utterances in conversation
        :return:
        """
        if len(labels) != number_of_utter:
            return False
        return True

    @classmethod
    def run_process(cls,
                    conv_str: str,
                    number_of_utter: int,
                    number_request: int = NUMBER_RESULT,
                    tool: enum.Enum = LLMsCompletionService.Tools.FARAROOM,
                    empathy_key_name: str = 'Empathy',
                    reasons_key_name: str = 'empathy_reasons',
                    percent_key_name: str = 'empathy_percents'):
        """
        this function plays as management of whole process:
            1. send request to get llms responses for conv_str
            2. extracted info from each response
            3. check if the response is incomplete or empthy to send request and extract again
            4. aggregate the result
            5. return all result

        :param conv_str: str of conversation
        :param number_of_utter: number of utterances in conversation
        :param number_request: number of requests for each conversation
        :param tool: which tool do you want to use for completion task?
        :param empathy_key_name: the key name of empathy_label. use for result
        :param reasons_key_name: the key name of reasons. use for result
        :param percent_key_name: the key name of percents. use for result
        :return:
        """
        all_labels, all_reasons = list(), list()
        incomplete_count = 0

        while len(all_labels) == number_request:
            responses = cls.get_response(conv_str=conv_str, tool=tool,
                                         num_requests=number_request if incomplete_count == 0 else incomplete_count)
            incomplete_count = 0
            for response in responses:
                labels, reasons = cls.extract_llms_response_info(response)
                if cls.is_complete(labels=labels, number_of_utter=number_of_utter):
                    all_labels.append(labels)
                    all_reasons.append(reasons)
                else:
                    incomplete_count += 1

        return cls.aggregate_responses(all_req_labels=all_labels, all_req_reasons=all_reasons,
                                       number_of_utter=number_of_utter, empathy_key_name=empathy_key_name,
                                       percent_key_name=percent_key_name, reasons_key_name=reasons_key_name)

