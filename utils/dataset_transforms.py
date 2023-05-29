import numpy as np
import torch


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
        texts = sample[0][0] if self.have_label else sample

        # value of each (row, col) can be list type or str type
        cleaned_result = self._clean_list_of_texts(texts) if isinstance(texts, list) else self._clean_single_text(texts)

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

    def __call__(self, sample):
        """
        :param sample:
        :return:
        """
        texts, target = sample
        texts = texts[0]

        conversation = str()
        for text in texts:
            conversation += f"{self.SPECIAL_TOKEN_START_UTTERANCE} {text} {self.SPECIAL_TOKEN_END_UTTERANCE} "
        return np.array([conversation]), target


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


class ToTensor:
    """
    Convert ndarrays to Tensors
    """

    def __call__(self, sample):
        return tuple(torch.from_numpy(np.array(each)) for each in sample)
