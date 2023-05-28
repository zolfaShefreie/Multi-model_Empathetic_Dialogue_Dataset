import yt_dlp as youtube_dl


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
        download_func(urls=urls, file_path=file_path, file_format=file_format)

    @staticmethod
    def _download_youtube(urls: list, file_path: str, only_audio=True, file_format="wav"):
        """

        :param urls: 
        :param file_path: 
        :param only_audio: 
        :param format: 
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
            'outtmpl': f"{file_path}.{file_format}",
            'noplaylist': True,
            'quiet': True,
            'no-warnings': True
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
