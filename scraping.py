from requests import get
from time import sleep
from random import randint
from bs4 import BeautifulSoup



pages = [str(i) for i in range(2,10)]
stars = '4'
file = 'print.txt'

textfile = open("print.txt", 'w')
t = 0
for page in pages:
    response = get('https://dk.trustpilot.com/review/www.farfetch.com?languages=en&page=' + page + '&stars=' + stars)
    
    sleep(randint(8,15))

    html_soup = BeautifulSoup(response.text, 'html.parser')

    review_container = html_soup.find_all('div', class_='paper_paper__1PY90 paper_square__lJX8a card_card__lQWDv card_noPadding__D8PcU styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')
    
    for i in range(len(review_container)):
        first_review = review_container[i]
        try:
            text = first_review.p.text + '\n' + '\n'
            if text[0:3]=='Læs':
                raise AttributeError
            t+=1
            #title = first_review.h2.a.text + '\n' + '\n'
            textfile.write(str(t) + '\n' + '\n')
            textfile.write(text)
            
            print(t)
        except AttributeError:
            print("no text")


textfile.close()
print("Done.")
