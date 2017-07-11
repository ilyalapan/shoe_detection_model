from bs4 import BeautifulSoup
import requests
from selenium import webdriver

from PIL import Image
import io
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import os
import sys

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
browser = webdriver.Chrome('driver/chromedriver')
abs_path_to_data = sys.argv[1] + '/'
print(abs_path_to_data)

def parse_waterfal_page(url,page):
    print('\n *Acessing Waterfall Webpage')
    r = requests.get(url.format(page))
    r.status_code
    html_doc = r.text
    soup = BeautifulSoup(html_doc, 'html.parser')
    links = soup.findAll("a", {"class": "product product-link "})
    for link in links:
        get_images_of_shoes(link['href'])

def get_images_of_shoes(link):
    global browser
    global abs_path_to_data
    browser.get(link)
    title = browser.title.lower().replace(' ','_').replace('/','_')
    path = abs_path_to_data + title
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Already exists, returning: ' + path)
        return
    print(title)
    contents = browser.find_elements_by_class_name('img')
    if len(contents) == 0:
        print(link)
        print('No contents found')
        raise
    for i in range(len(contents)):
        src = contents[i].get_attribute('src')
        print(src)
        r = requests.get(src)
        try:
            image = Image.open(io.BytesIO(r.content))
            save_path = path + '/' + str(i) + '.jpg'
            image.save(save_path)
            print('Saved: ' + save_path)
        except:
            print('COULDNT SAVE A FILE!!!')

def cycle_page_url(url, page_num):
    for page in range(page_num):
        parse_waterfal_page(url, page)

if not os.path.exists(abs_path_to_data):
    print(abs_path_to_data + '  does not exist')
    raise
else:
    print('Already exists: ' + abs_path_to_data)


#cycle_page_url('http://www.asos.com/men/shoes-boots-trainers/cat/?cid=4209&pge={0}&pgesize=204',12)
cycle_page_url('http://www.asos.com/women/shoes/cat/?cid=4172&pge={0}&pgesize=204',11)


