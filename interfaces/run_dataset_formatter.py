import argparse

from dataset_format_converter import dataset_process


class RunningFormatterInterface:

    # keys are the name of arguments that it must be unique
    ARGUMENTS = {
        '--dataset_name': {
            # todo make sure for consider all choicess
            'help': 'name of dataset, can be meld, dailytalk, annomi',
            'choices': ['meld', 'dailytalk', 'annomi'],
            'required': True
        },
        '--dataset_dir': {
            'help': "the directory of dataset.\n"
                    "WARNING: this program doesn't support zip files so do make sure data and files are unzip \n"
                    "WARNING: directory must be the path where data files are on it",
            'required': True
        },
        '--save_at_dir': {
            'help': "the directory that you want to save formatted data on it",
            'request': True
        },
    }

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        """
        add self.arguments to self.parser
        :return:
        """
        for argument_name, options in self.ARGUMENTS.items():
            self.parser.add_argument(argument_name, **options)


if __name__ == "__main__":
    pass