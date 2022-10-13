import requests
import bs4
from slugify import slugify
import os


def crawl(url):
    print("crawling from ", url, "...")
    domain = url.split("https://")[-1].split("/")[0].split(".")[1]
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"})
    if html.status_code != 200:
        print(html.status_code)
    soup = bs4.BeautifulSoup(html.content, "lxml")
    links = set(soup.find_all('a', href=True))

    i = 0

    foldername = domain
    if not(os.path.isdir(foldername)):
        os.mkdir(foldername)
    print("len of links: ", len(links))

    for link in links:
        sub_url = link['href']
        page_name = link.string
        if i>len(links):
          break
        else:
          i += 1
        if 'abs/' in sub_url and '#' not in sub_url:
            try:
                if page_name:
                    page = requests.get(url.split('list')[0] + sub_url[1:], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"})
                    #filename = slugify(page_name).lower() + '.html'
                    if page.status_code != 200:
                        print(page.status_code)
                        pass
                    path = foldername + '/' + sub_url.split('/')[2] + '.html'
                    html_path_list.append(sub_url.split('/')[2] + '.html')
                    with open(path, 'wb') as f:
                        f.write(page.content)
            except:
                pass
    print("html_path_list len: ", len(html_path_list))

#https://arxiv.org/list/cs/2209?skip=2000&show=2000        #skip 부터 show개 만큼
urls = ["https://export.arxiv.org/list/cs/2207?skip=" + str(i) + "&show=2000" for i in range(0,8000,2000)]
print(urls)
html_path_list = []

for url in urls:
    crawl(url)
print(len(html_path_list))
with open(os.getcwd() + '\\html_path_list.txt', 'wb') as f:
    for i in range(len(html_path_list)):
        f.write((html_path_list[i] + '\n').encode('utf-8'))



#크롤링 걸리는 시간 월단위로 5:20~7:26 약 2시간