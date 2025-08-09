import os
import urllib.request
import urllib.error
import ssl
from urllib.parse import urlparse
import time
import sys
from pathlib import Path


def read_urls_from_file(filename):

    urls = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('http'):
                    urls.append(line)
    except FileNotFoundError:
        print(f"Error：can not find {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error when reading：{e}")
        sys.exit(1)

    return urls


def sample_urls(urls, sample_size=100):

    total_urls = len(urls)
    if total_urls <= sample_size:
        print(f"warning：{total_urls} URLs in total，gonna download all")
        return urls

    interval = total_urls / sample_size
    sampled_urls = []

    for i in range(sample_size):
        index = int(i * interval)
        if index < total_urls:
            sampled_urls.append(urls[index])

    print(f"from all {total_urls} URLs, sampling {len(sampled_urls)} URLs")
    return sampled_urls


def download_file(url, download_dir="downloads"):

    try:
        # create dir
        os.makedirs(download_dir, exist_ok=True)

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filepath = os.path.join(download_dir, filename)

        if os.path.exists(filepath):
            print(f"File existed，jump：{filename}")
            return True

        print(f"Downloading：{filename}...")

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        urllib.request.install_opener(opener)

        # Download File
        urllib.request.urlretrieve(url, filepath)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"successfully download：{filename} ({os.path.getsize(filepath):,} bytes)")
            return True
        else:
            print(f"fail to download：{filename}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    except Exception as e:
        print(f"Error when downloading {filename} ：{e}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return False


def main():
    # config
    INPUT_FILE = "/home/fisher/PycharmProjects/AutoCompose/ml-cubifyanything/data/train.txt"
    SAMPLE_SIZE = 500
    DOWNLOAD_DIR = "/data/CA-1M-download"


    print("=" * 50)
    print("ca1m Dataset Downloader")
    print("=" * 50)

    # Reading URL
    print(f"reading url from{INPUT_FILE} ...")
    urls = read_urls_from_file(INPUT_FILE)

    # Sampling
    print(f"sampling，target num：{SAMPLE_SIZE}")
    sampled_urls = sample_urls(urls, SAMPLE_SIZE)

    # create download dir
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"download dir：{os.path.abspath(DOWNLOAD_DIR)}")

    print("\nDownloading sampled urls...")
    success_count = 0
    failed_count = 0

    for i, url in enumerate(sampled_urls, 1):
        print(f"\n[{i}/{len(sampled_urls)}] ", end="")
        if download_file(url, DOWNLOAD_DIR):
            success_count += 1
        else:
            failed_count += 1

        time.sleep(0.1)


    print("\n" + "=" * 50)
    print("download completely！")
    print(f"successfully saved：{success_count} files")
    print(f"fail to save：{failed_count} files")
    print(f"download dir：{os.path.abspath(DOWNLOAD_DIR)}")


    if success_count > 0:
        print("\nsaved files list：")
        downloaded_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.tar')]
        for filename in sorted(downloaded_files):
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            size = os.path.getsize(filepath)
            print(f"  {filename} ({size:,} bytes)")


if __name__ == "__main__":
    main()