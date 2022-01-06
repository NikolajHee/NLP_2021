from requests import get

url = 'https://dk.trustpilot.com/review/www.farfetch.com?languages=en&page=2&stars=4'

response = get(url)

from bs4 import BeautifulSoup

html_soup = BeautifulSoup(response.text, 'html.parser')

review_container = html_soup.find_all('div', class_='paper_paper__1PY90 paper_square__lJX8a card_card__lQWDv card_noPadding__D8PcU styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')

textfile = open("print.txt", 'w')

for i in range(len(review_container)):
    first_review = review_container[i]
    print("title:", first_review.h2.a.text)
    try:
        text = first_review.p.text + '\n' + '\n'
        title = first_review.h2.a.text + '\n' + '\n'
        textfile.write(title)
        textfile.write(text)
        print('success')
    except AttributeError:
        print("no text")

