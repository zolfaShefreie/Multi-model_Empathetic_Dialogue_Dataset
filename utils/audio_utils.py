from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation
import moviepy.editor as mp

from settings import ASR_MODEL_NAME


class AudioModule:
    ASR_MODEL = EncoderDecoderASR.from_hparams(source=ASR_MODEL_NAME)
    ALIGNER = CTCSegmentation(ASR_MODEL, kaldi_style_text=False)

    @classmethod
    def extract_audio_from_video(cls, video_path: str, saved_path: str):
        """
        extract audio from video
        :param video_path: path of video
        :param saved_path: where the audio must be saved
        :return:
        """
        mp.VideoFileClip(video_path).audio.write_audiofile(saved_path)

    @classmethod
    def convert_to_wav(cls):
        pass

    @classmethod
    def get_timestamp(cls, file_path: str, utterances: list) -> list:
        """
        get timestamps of utterances
        :param file_path: path of audio file
        :param utterances: list of utterance
        WARNING:
                for better performance all of utterances in audio must be included
        :return: list of (start_time, end_time)
        """
        segments = cls.ALIGNER(file_path, utterances, name=file_path.strip("/")[-1])
        return [(each[0], each[1]) for each in segments.segments]

    @classmethod
    def segment_audio(cls, file_path: str, utterances: list, save_dir: str):
        pass
