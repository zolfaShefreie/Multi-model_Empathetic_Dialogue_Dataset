import argparse
import os

from dataset_format_converter import dataset_process
from utils.interface import BaseInterface


class RunningFormatterInterface(BaseInterface):

    DESCRIPTION = "This interface runs the formatter stages for specific dataset"

    # keys are the name of arguments that it must be unique
    ARGUMENTS = {
        'dataset_name': {
            # todo make sure for consider all choicess
            'help': 'name of dataset, can be meld, dailytalk, annomi',
            'choices': ['meld', 'dailytalk', 'annomi'],
            'required': True
        },
        'dataset_dir': {
            'help': "the directory of dataset.\n"
                    "WARNING: this program doesn't support zip files so do make sure data and files are unzip \n"
                    "WARNING: directory must be the path where data files are on it",
            'required': True
        },
        'save_at_dir': {
            'help': "the directory that you want to save formatted data on it",
            'required': True
        },

        'start_stage': {
            'help': 'running stages form this stage',
            'required': False,
            'default': None
        },

        'stop_stage': {
            'help': 'stop running stages at this stage',
            'required': False,
            'default': None
        }
    }

    FORMATTER = {
        'meld': dataset_process.MELDDatasetFormatter,
        'dailytalk': dataset_process.DailyTalkDatasetFormatter,
        'annomi': dataset_process.AnnoMIDatasetFormatter
    }

    def validate_dataset_dir(self, value):
        if not os.path.exists(value):
            raise Exception("path doesn't exists")
        return value

    def _run_main_process(self):
        """
        get formatter class and make new object of it and run the formatter on specific stages
        :return:
        """
        formatter_class = self.FORMATTER[self.dataset_name]
        formatter_obj = formatter_class(dataset_dir=self.dataset_dir, save_dir=self.save_at_dir)
        try:
            formatter_obj.running_process(start_stage=self.start_stage, stop_stage=self.stop_stage)
        except Exception as e:
            print(f"____________________________________\n"
                  f"AN ERROR OCCURRED:\n"
                  f"{str(e)}\n"
                  f"____________________________________")


if __name__ == "__main__":
    RunningFormatterInterface().run()
