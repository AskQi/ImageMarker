from icrawler.builtin import GoogleImageCrawler
from icrawler.downloader import ImageDownloader


class MyImageDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        # 生成文件名：使用关键词+原文件名
        filename = f"{self.keyword.replace(' ', '_')}_{super().get_filename(task, default_ext)}"
        return filename


def download_images(keyword, limit=5, download_path='data'):
    google_crawler = GoogleImageCrawler(downloader_cls=MyImageDownloader,storage={'root_dir': download_path})
    google_crawler.downloader.keyword = keyword
    google_crawler.crawl(keyword=keyword, max_num=limit)

download_images('Asian Portrait Photography',10,'data')
download_images('Attractive Asian Faces',10,'data')
download_images('Sleeping Asian Model Photography',10,'data')