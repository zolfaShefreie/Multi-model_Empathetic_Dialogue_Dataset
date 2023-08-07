import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)


RAW_DATASET_PATH = env("RAW_DATASET_PATH", default="./raw_dataset")

# Classifier File Path
EMPATHY_KIND_MODEL_FILE_PATH = env("EMPATHY_KIND_MODEL_FILE_PATH", default="./")
EMPATHY_EXIST_MODEL_FILE_PATH = env("EMPATHY_EXIST_MODEL_FILE_PATH", default="./")

# use for writerLoaderHandler
PREFIX_MID_PROCESS_DIR = "./middle_pipeline_stage"
PREFIX_MID_PROCESS_CACHE_DIR = "./.cache_stages"

# use for AudioModule
# about the model you can read here https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/
ASR_MODEL_NAME = "speechbrain/asr-conformer-transformerlm-librispeech"
