import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)


RAW_DATASET_PATH = env("RAW_DATASET_PATH", default="./raw_dataset")
PREFIX_CLASSIFIER_DIR = env("PREFIX_CLASSIFIER_DIR", default="./")

PREFIX_MID_PROCESS_DIR = env("PREFIX_MID_PROCESS_DIR", default="./middle_pipeline_stage")
