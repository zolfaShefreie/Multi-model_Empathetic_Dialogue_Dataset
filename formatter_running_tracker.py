import os

from dataset_format_converter import dataset_process
from utils.interface import BaseInterface


class TrackerFormatterInterface(BaseInterface):

    DESCRIPTION = "This interface shows which stages have already been run for specific dataset"

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
        stage_list = formatter_class.stage_tracker()
        if not stage_list:
            print(f"____________________________________\n"
                  f"Any stages of {self.dataset_name} dataset has not been run\n"
                  f"____________________________________")
        else:
            print(f"____________________________________\n"
                  f"Run Stages: {stage_list}\n"
                  f"____________________________________")


if __name__ == "__main__":
    TrackerFormatterInterface().run()
