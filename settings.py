import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)


RAW_DATASET_PATH = env("RAW_DATASET_PATH", default="./raw_dataset")
PREFIX_CLASSIFIER_DIR = env("PREFIX_CLASSIFIER_DIR", default="./")

# use for writerLoaderHandler
PREFIX_MID_PROCESS_DIR = "./middle_pipeline_stage"
PREFIX_MID_PROCESS_CACHE_DIR = "./.cache_stages"

# use for AudioModule
# about the model you can read here https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/
ASR_MODEL_NAME = "speechbrain/asr-transformer-transformerlm-librispeech"
