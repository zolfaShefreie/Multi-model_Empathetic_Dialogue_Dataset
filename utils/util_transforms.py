import numpy as np
import torch


class Pipeline:

    def __init__(self, functions: list):
        self.functions = functions

    def __call__(self, data):
        for func in self.functions:
            data = func(data)
        return data


class TextCleaner:
    PUNC = '''!()-[]{.};:'"\,<>/?@#$%^&*_~`|’“”…—–'''

    def __init__(self, have_label=True):
        self.have_label = have_label

    @classmethod
    def _clean_single_text(cls, text: str) -> str:
        """
        return cleaned text
        :param text:
        :return:
        """
        text = text.lower()
        for each in cls.PUNC:
            text = text.replace(each, ' ')

        return text

    @classmethod
    def _clean_list_of_texts(cls, texts: list) -> list:
        """
        clean list of texts
        :param texts:
        :return:
        """
        return [cls._clean_single_text(text) for text in texts]

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        texts = sample[0][0] if self.have_label else sample[0]

        # value of each (row, col) can be list type or str type
        cleaned_result = self._clean_list_of_texts(texts) if isinstance(texts, list) or isinstance(texts, np.ndarray) \
            else self._clean_single_text(texts)

        if self.have_label:
            return np.array([cleaned_result]), sample[-1]
        else:
            return np.array([cleaned_result])


class Tokenizer:

    def __init__(self, tokenizer, have_label=True, max_len=128, new_special_tokens=None):
        """
        :param tokenizer:
        :param have_label:
        :param max_len:
        :param new_special_tokens:
        """
        self.tokenizer = tokenizer

        if new_special_tokens:
            tokenizer.add_special_tokens(new_special_tokens)

        self.have_label = have_label
        self.MAX_LEN = max_len

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """

        text = sample[0] if self.have_label else sample

        inputs = self.tokenizer.encode_plus(text[0],
                                            add_special_tokens=True,
                                            max_length=self.MAX_LEN,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            truncation=True)

        if self.have_label:
            return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], sample[-1]
        else:
            return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']


class ConversationFormatter:
    """
    join utterance with special tokens
    """

    SPECIAL_TOKEN_START_UTTERANCE = "<BOU>"
    SPECIAL_TOKEN_END_UTTERANCE = "<EOU>"

    def __init__(self, have_label=True):
        self.have_label = have_label

    def __call__(self, sample):
        """
        :param sample:
        :return:
        """
        texts = sample[0] if self.have_label else sample

        conversation = str()
        for text in texts:
            conversation += f"{self.SPECIAL_TOKEN_START_UTTERANCE} {text} {self.SPECIAL_TOKEN_END_UTTERANCE} "

        if self.have_label:
            return np.array([conversation]), sample[-1]
        else:
            return np.array([conversation])


class ToTensor:
    """
    Convert ndarrays to Tensors
    """

    def __call__(self, sample):
        return tuple(torch.from_numpy(np.array(each)) for each in sample)


class AddBatchDimension:
    """
    add batch dimension to one sample
    """

    def __call__(self, sample):
        return [torch.unsqueeze(x, 0) for x in sample]


class ConvertInputToDict:

    def __init__(self, dict_meta_data: dict):
        """

        :param dict_meta_data: meta data about how the output must be look like
         (key, index in sample)
        """
        self.dict_meta_data = dict_meta_data

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        return {key: sample[index] for key, index in self.dict_meta_data.items()}


class OneHotLabel:
    """
    this class is used for label with multi classes
    """

    def __init__(self, num_classes):
        """

        :param num_classes: number of classes
        """
        self.num_classes = num_classes

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        target = sample[-1]
        target = torch.squeeze(torch.nn.functional.one_hot(target, num_classes=self.num_classes), dim=0)
        sample = list(sample[:-1]) + [target]
        return tuple(sample)


# output transformer


class DelBatchDimension:
    """
    delete batch dimension to one output
    """

    def __call__(self, model_output):
        return torch.squeeze(model_output, 0)


class IntegerConverterWithThreshold:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, model_output):
        return (model_output > self.threshold).float()


class IntegerConverterWithIndex:

    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, model_output):
        return torch.argmax(model_output, dim=self.dim)
