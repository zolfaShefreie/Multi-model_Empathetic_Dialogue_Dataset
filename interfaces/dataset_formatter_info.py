
from dataset_format_converter import dataset_process
from utils.interface import BaseInterface


class FormatterInfoInterface(BaseInterface):
    """
    this interface show with stages have already been run
    """

    # keys are the name of arguments that it must be unique
    ARGUMENTS = {
        'dataset_name': {
            # todo make sure for consider all choicess
            'help': 'name of dataset, can be meld, dailytalk, annomi',
            'choices': ['meld', 'dailytalk', 'annomi'],
            'required': True
        },
    }

    FORMATTER = {
        'meld': dataset_process.MELDDatasetFormatter,
        'dailytalk': dataset_process.DailyTalkDatasetFormatter,
        'annomi': dataset_process.AnnoMIDatasetFormatter
    }

    def _run_main_process(self):
        """
        get formatter class and run stage tracker to get run stages
        :return:
        """
        formatter_class = self.FORMATTER[self.dataset_name]
        stage_list = formatter_class.SEQ_STAGE
        print(f"____________________________________\n"
              f"Stages:\n")
        for stage in stage_list:
            print(f"Stage: {stage}\n Description: {getattr(formatter_class, stage).__doc__.strip().split(':')[0]}\n")
        print(f"____________________________________")


if __name__ == "__main__":
    FormatterInfoInterface().run()