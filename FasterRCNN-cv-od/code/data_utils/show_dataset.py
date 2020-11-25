# %%
from bs4 import BeautifulSoup
xml_path = "../../data/VOCdevkit/VOC2007/Annotations/000001.xml"
with open(xml_path) as f:
    soup = BeautifulSoup(f, 'xml')
    # print(soup)
    print(soup.find('filename').text)
    floder = soup.floder.text

# %%
