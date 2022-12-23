import requests
from bs4 import BeautifulSoup
import json
import os
import subprocess


def get_target(keyword_, page_, save_name_):
    print("正在获取关于{}的视频，共{}页,保存路径为D:/bilibili_video/{}".format(keyword_, page_, save_name_))
    bvs = []
    for page_id in range(1, page_ + 1):
        r = requests.get("https://search.bilibili.com/all"
                         "?vt=00210960"
                         "&keyword={}"
                         "&from_source=webtop_search"
                         "&spm_id_from=333.1007&search_source=5"
                         "&page={}"  # 页号
                         "&o={}".format(keyword_, page_id, page_id * 30))  # o 每多一页加30
        soup = BeautifulSoup(r.text)

        res_ = soup.find_all(href=True, class_="img-anchor")
        for i in res_:
            bvs.append(i.attrs["href"][25:37])

    for bv in bvs:
        url = "https://www.bilibili.com/video/{}/".format(bv)

        v_page = requests.get(url)
        soup2 = BeautifulSoup(v_page.text)

        video_url = json.loads(soup2.find("head").find_all("script")[2].
                               text[20:])["data"]["dash"]["video"][0]['baseUrl']

        audio_url = json.loads(soup2.find("head").find_all("script")[2].
                               text[20:])["data"]["dash"]["audio"][0]['baseUrl']

        p = 1

        get(url, video_url, audio_url, p, bv)


def file_download(home_url, url, name, session=requests.session()):
    # 添加请求头键值对,写上 referer:请求来源

    headers.update({'Referer': home_url})

    # 发送option请求服务器分配资源

    session.options(url=url, headers=headers, verify=False)

    # 指定每次下载1M的数据

    begin = 0

    end = 1024 * 512 - 1

    flag = 0

    while True:

        # 添加请求头键值对,写上 range:请求字节范围

        headers.update({'Range': 'bytes=' + str(begin) + '-' + str(end)})

        # 获取视频分片

        res = session.get(url=url, headers=headers, verify=False)

        if res.status_code != 416:

            # 响应码不为为416时有数据

            begin = end + 1

            end = end + 1024 * 512

        else:

            headers.update({'Range': str(end + 1) + '-'})

            res = session.get(url=url, headers=headers, verify=False)

            flag = 1

        with open(name.encode("utf-8").decode("utf-8"), 'ab') as fp:

            fp.write(res.content)

            fp.flush()

        # data=data+res.content

        if flag == 1:
            fp.close()

            break


def get(url, video_url, audio_url, p, bv):
    session = requests.session()

    dirname = "D:/bilibili_video/{}/".format(saveName).encode("utf-8").decode("utf-8")

    if not os.path.exists(dirname):
        os.makedirs(dirname)

        print('文件夹创建成功!')

    name = bv + "-" + str(p)  # 获取每一集的名称

    # 下载视频和音频

    print('正在下载 "' + name + '" 的视频····')

    file_download(home_url=url, url=video_url, name=dirname + name + '_Video.mp4', session=session)

    print('正在下载 "' + name + '" 的音频····')

    file_download(home_url=url, url=audio_url, name=dirname + name + '_Audio.mp4', session=session)

    print(' "' + name + '" 下载完成！')

    print("合并音频")

    combine_video_audio(dirname + name + '_Video.mp4', dirname + name + '_Audio.mp4', dirname + name + '.mp4')


def combine_video_audio(videopath, audiopath, outpath):
    subprocess.call((
                                "C:/ffmpeg/ffmpeg-2022-12-19-git-48d5aecfc4-essentials_build/bin/ffmpeg -i " + videopath + " -i " + audiopath + " -c copy " + outpath).encode(
        "utf-8").decode("utf-8"), shell=True)

    os.remove(videopath)

    os.remove(audiopath)


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()

    headers = {

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/80.0.3970.5 Safari/537.36',

        'Refer'

        'er': 'https://www.bilibili.com/'

    }

    keyword = input("请输入要搜索的关键词：")
    page = int(input("请输入要爬取的页数："))
    saveName = input("请输入要保存的文件名：")
    get_target(keyword, page, saveName)
