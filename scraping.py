from requests import get
from time import sleep
from random import randint
from bs4 import BeautifulSoup
from IPython.core.display import clear_output



pages = [str(i) for i in range(2,25)]
stars = '1'

textfile = open('Auto/auto_'+stars+'stjerner.txt', 'w')
t = 0
p = 0
for page in pages:
    p+=1
    response = get('https://dk.trustpilot.com/review/www.farfetch.com?languages=en&page=' + page + '&stars=' + stars)
    
    sleep(randint(10,17))

    html_soup = BeautifulSoup(response.text, 'html.parser')

    review_container = html_soup.find_all('div', class_='paper_paper__1PY90 paper_square__lJX8a card_card__lQWDv card_noPadding__D8PcU styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')
    
    for i in range(len(review_container)):
        first_review = review_container[i]
        try:
            text = first_review.p.text + '\n' + '\n'
            if text[0:3]=='Læs':
                raise AttributeError
            elif text[0:23]=='Besvarelse fra Farfetch':
                raise AttributeError
            t+=1
            #title = first_review.h2.a.text + '\n' + '\n'
            textfile.write(str(t) + '\n' + '\n')
            textfile.write(text)
            
            mes = str(t)
        except AttributeError:
            mes = 'no text'
        finally:
            print("Line: {}; Page: {};".format(mes,p))
            clear_output(wait = True)

print("DONE.")

textfile.close()


#%%

from requests import get
from time import sleep
from random import randint
from bs4 import BeautifulSoup
from IPython.core.display import clear_output


page = '2'
stars = '1'
response = get('https://dk.trustpilot.com/review/www.farfetch.com?languages=en&page=' + page + '&stars=' + stars)
html_soup = BeautifulSoup(response.text, 'html.parser')

review_container = html_soup.find_all('div', class_='paper_paper__1PY90 paper_square__lJX8a card_card__lQWDv card_noPadding__D8PcU styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')

# %%
