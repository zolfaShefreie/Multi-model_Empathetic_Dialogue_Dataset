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

# tools for LLMs
# API key or token configs

FARAROOM_AUTH_CONFIG = {
    'FARAROOM_TOKEN': env("FARAROOM_TOKEN", default=None),
    'FARAROOM_META': env("FARAROOM_META", default=None),
    'FARAROOM_SOURCE': env("FARAROOM_META", default=None),
    'FARAROOM_TASK_NAME': env("FARAROOM_TASK_NAME", default=None),
}

OPENAI_API_KEY = env("OPENAI_API_KEY", default=None)
OPENAI_MODEL = 'gpt-3.5-turbo-1106'

TOGETHER_API_KEY = env("TOGETHER_API_KEY", default=None)
TOGETHER_MODEL = "togethercomputer/CodeLlama-34b-Instruct"


HUGGING_FACE_REPO_NAME = env("HUGGING_FACE_REPO_NAME", default="")
HUGGING_FACE_IS_PRIVATE = env("HUGGING_FACE_IS_PRIVATE", default=True)
HUGGING_FACE_TOKEN = env("HUGGING_FACE_TOKEN", default=None)
