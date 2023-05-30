import torch


class BaseClassifier:
    CONFIG_VALID_KEYS = {'checkpoint_path', 'model_class', 'pre_process_data_pipeline', 'end_apply_result_pipeline'}

    @classmethod
    def _validate_classifier_config(cls, classifier_config: dict) -> dict:
        """
        :param classifier_config: check to have all necessary keys
        """
        if cls.CONFIG_VALID_KEYS ^ set(classifier_config.keys()):
            raise Exception("doesn't contain some configs or include invalid keys")
        return classifier_config

    def __init__(self, classifier_config: dict):
        """
        :param classifier_config:
        """
        classifier_config = self._validate_classifier_config(classifier_config=classifier_config)

        self.pre_process_data_pipeline = classifier_config['pre_process_data_pipeline']
        self.end_apply_result_pipeline = classifier_config['end_apply_result_pipeline']

        # load model
        self.model_class = classifier_config['model_class']
        self.model = self.model_class.load_from_checkpoint(classifier_config['checkpoint_path'],
                                                           map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                                     else 'cpu'))

    def predict(self, data):
        """
        :param data:
        :return:
        """
        processed_data = self.pre_process_data_pipeline(data)
        model_output = self.model(processed_data)
        return self.end_apply_result_pipeline(model_output)

    def __call__(self, data):
        return self.predict(data)
