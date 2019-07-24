import bs4 as bs
import lxml
import urllib.request
import pandas as pd

sauce = urllib.request.urlopen("https://karki23.github.io/Weather-Data/assignment.html")
srccode= bs.BeautifulSoup(sauce,'lxml')
al=srccode.find_all('a')
std = "https://karki23.github.io/Weather-Data/"
flag=True
csv=[]

for i in al:
    city=[]
    link_obj = urllib.request.urlopen(std+i.get('href'))
    link_src = bs.BeautifulSoup(link_obj,'lxml')
    if(flag):
        header=[j.text for j in link_src.find_all('th')]
        flag=False
    rows=link_src.find_all('tr')
    for j in range(1,len(rows)):
        cells=[k.text for k in rows[j].find_all('td')]
        csv.append(cells)
        city.append(cells)
    citydf = pd.DataFrame(city ,columns=header)
    citydf.to_csv(i.text+".csv",index=False)

dataframe = pd.DataFrame(csv ,columns=header)
dataframe.to_csv("dataset.csv",index=False)