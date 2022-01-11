from requests import get
from time import sleep
from random import randint
from bs4 import BeautifulSoup
from IPython.core.display import clear_output
import csv





#textfile = open('Auto/auto_'+stars+'stjerner.txt', 'w')

f = open('Auto/data_set.csv','w')

writer = csv.writer(f)

header = ['Title','Data','Catagory']

writer.writerow(header)






stars = ['1','2','4','5']
max_pages = [25,20,20,25]
category = ['neg','neg','pos','pos']


for i in range(4):
    star = stars[i]
    max_page = max_pages[i]
    pages = [str(j) for j in range(2,max_page)]
    
    
    t = 0
    p = 0
    for page in pages:
        p+=1
        response = get('https://dk.trustpilot.com/review/www.farfetch.com?languages=en&page=' + page + '&stars=' + stars[i])
        
        sleep(randint(10,17))

        html_soup = BeautifulSoup(response.text, 'html.parser')

        review_container = html_soup.find_all('div', class_='paper_paper__1PY90 paper_square__lJX8a card_card__lQWDv card_noPadding__D8PcU styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')
        
        for j in range(len(review_container)):
            iteration = []
            first_review = review_container[j]
            try:
                text = first_review.p.text + '\n' + '\n'
                if text[0:3]=='Læs':
                    raise AttributeError
                elif text[0:23]=='Besvarelse fra Farfetch':
                    raise AttributeError
                t+=1
                title = first_review.h2.a.text + '\n' + '\n'
                mes = str(t)
                
                iteration = [title, text, category[i]]
                
                writer.writerow(iteration)
                
                
            except AttributeError:
                mes = 'no text'
            finally:
                print("Line: {}; Page: {};".format(mes,p))
                clear_output(wait = True)


print("DONE.")





