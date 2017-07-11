from bs4 import BeautifulSoup
import requests
from selenium import webdriver

from PIL import Image
import io
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import os
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
browser = webdriver.Chrome('driver/chromedriver')


def parse_waterfal_page(url):
    print('\n *Acessing Waterfall Webpage')
    r = requests.get(url)
    r.status_code
    html_doc = r.text
    soup = BeautifulSoup(html_doc, 'html.parser')
    elements = soup.findAll("div", {"class": "item threeacross"})
    for element in elements:
        link = 'https://marketplace.asos.com'+element.findChildren('a')[0]['href']
        get_images_of_shoes(link)


def get_images_of_shoes(link):
    global browser
    browser.get(link)
    title = browser.title.lower().replace(' ','_')
    path = '/data/' + title
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Already exists, returning: ' + path)
        return
    content = browser.find_element_by_class_name('c1').get_attribute('innerHTML')
    soup = BeautifulSoup(content, 'html.parser')
    images = soup.findAll('img')
    for i in range(len(images)):
        src = images[i]['src']
        src = src.replace('small', 'large')
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
    for page in range(27,page_num):
        url_page = url.format(page)
        print(url_page)
        parse_waterfal_page(url_page)

if not os.path.exists('data'):
    os.makedirs('data')
else:
    print('Already exists: ' + 'data')


#cycle_page_url('https://marketplace.asos.com/men/shoes?ctaref=mktp%7Cmw%7Cnav%7Cshoes&pgno={0}',4)
cycle_page_url('https://marketplace.asos.com/women/shoes?ctaref=mktp|ww|nav|shoes&pgno={0}',56)


