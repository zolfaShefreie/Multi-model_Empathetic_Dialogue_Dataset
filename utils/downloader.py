import yt_dlp as youtube_dl
import os


class Downloader:
    """
    example:
        Downloader.download(download_type='youtube', urls=['https://www.youtube.com/watch?v=PaSKcfTmFEk'],
                            file_path="./%(id)s", file_format="wav")
    """
    VALID_URLS = ['youtube', ]

    @classmethod
    def download(cls, download_type: str, urls: list, file_path: str, file_format: str = None):
        """

        :param file_format:
        :param file_path:
        :param download_type:
        :param urls:
        :return:
        """

        if download_type not in cls.VALID_URLS:
            raise Exception("download type not found")
        download_func = getattr(cls, f"_download_{download_type}")
        return download_func(urls=urls, file_path=file_path, file_format=file_format)

    @staticmethod
    def _download_youtube(urls: list, file_path: str, only_audio=True, file_format="wav"):
        """
        download multi file from youtube
        :param urls: list of url
        :param file_path: it can be like youtube_dl format or just path
        :param only_audio: option to save audio or video
        :param file_format: format of file
        :return: 
        """""

        ydl_opts = {
            'force-ipv4': True,
            "external-downloader": "aria2c",
            "external-downloader-args": "-x 16 -s 16 -k 1M",
            'ignoreerrors': True,
            'format': 'bestaudio/best',
            'extractaudio': only_audio,
            'audioformat': file_format,
            'outtmpl': f"{file_path}",
            'noplaylist': True,
            'quiet': True,
            'no-warnings': True
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)

        return file_path if os.path.exists(file_path) else None
