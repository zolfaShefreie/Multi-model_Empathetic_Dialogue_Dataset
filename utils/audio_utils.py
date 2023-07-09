from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation
import moviepy.editor as mp
from pydub import AudioSegment

from settings import ASR_MODEL_NAME


class AudioModule:
    ASR_MODEL = EncoderDecoderASR.from_hparams(source=ASR_MODEL_NAME)
    ALIGNER = CTCSegmentation(ASR_MODEL, kaldi_style_text=False)

    @classmethod
    def extract_audio_from_video(cls, video_path: str, saved_path: str) -> str:
        """
        extract audio from video
        :param video_path: path of video
        :param saved_path: where the audio must be saved
        :return:
        """
        mp.VideoFileClip(video_path).audio.write_audiofile(saved_path)
        return saved_path

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
    def segment_audio(cls, file_path: str, utterances: list, prefix_name: str, save_dir: str,
                      first_utter_id: int = 0) -> list:
        """
        split audio to get audio of each utterance and save it with wav format
        :param file_path: path of audio file
        :param utterances: list of utterance
        :param prefix_name: prefix name of sub audios
        :param save_dir: dir that audios gonna be saved
        :param first_utter_id: the index of the first element of utterances(param) in the complete conversation
        :return: a list of sub audio path
        """
        if len(utterances) > 1:
            timestamps = cls.get_timestamp(file_path=file_path, utterances=utterances)
            audio = AudioSegment.from_wav(file_path)

            segments_path = list()

            for index, seg in enumerate(timestamps):
                # pydub works with milliseconds and speechBrain works with seconds
                start, end = seg[0] * 1000, seg[1] * 1000

                audio_chunk = audio[start:end]
                audio_chunk.export(f"{save_dir}/{prefix_name}_{index + first_utter_id}.wav", format="wav")
                segments_path.append(f"{save_dir}/{prefix_name}_{index + first_utter_id}.wav")

            return segments_path

        elif utterances == 1:
            return [file_path]

        else:
            return list()
