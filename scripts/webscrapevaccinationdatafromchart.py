from bs4 import BeautifulSoup
import requests



url1='https://yannickbochatay.github.io/covid-19-vaccination/build/'
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.text, 'html.parser')
print(soup1)


