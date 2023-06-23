import requests
from bs4 import BeautifulSoup

from utils import HEADERS

OUT_FILENAME = 'data-metadata/ru_words.txt'

wiki_link = "https://ru.wiktionary.org/wiki/"
top1_100 = "Приложение:Список_частотности_по_НКРЯ"
top101_1000 = "Приложение:Список_частотности_по_НКРЯ/101—1000"
top1001_10000 = "Приложение:Список_частотности_по_НКРЯ/1001—10_000"

top1_100_wiki_link = "https://ru.wiktionary.org/wiki/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5:%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%87%D0%B0%D1%81%D1%82%D0%BE%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8_%D0%BF%D0%BE_%D0%9D%D0%9A%D0%A0%D0%AF"
top101_1000_wiki_link = "https://ru.wiktionary.org/wiki/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5:%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%87%D0%B0%D1%81%D1%82%D0%BE%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8_%D0%BF%D0%BE_%D0%9D%D0%9A%D0%A0%D0%AF/101%E2%80%941000"
top1001_10000_wiki_link = "https://ru.wiktionary.org/wiki/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5:%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%87%D0%B0%D1%81%D1%82%D0%BE%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8_%D0%BF%D0%BE_%D0%9D%D0%9A%D0%A0%D0%AF/1001%E2%80%9410_000"
LINKS = (top1_100_wiki_link, top101_1000_wiki_link, top1001_10000_wiki_link)


def get_words_page(link):
    session = requests.session()
    response = session.get(link, headers=HEADERS)

    if response.status_code == 200:
        print(f"Successfully loaded `{requests.utils.unquote(link)}`")
    else:
        raise ValueError("Something went wrong while getting result")

    words = []

    soup = BeautifulSoup(response.text, 'html.parser')
    for list_element in soup.find(
            'div', id="mw-content-text").find('ol').find_all('li'):
        word = list_element.string
        if word is not None:
            words.append(word)

    print(f'Successfully parsed `{requests.utils.unquote(link)}`')
    return words


def get_words():
    words = []
    for link in LINKS:
        words.extend(get_words_page(link))
    return words


def delete_duplicates(words):
    return list(dict.fromkeys(words))


with open(OUT_FILENAME, 'w', encoding='utf-8') as out_f:
    words = get_words()
    print(f"Before removing duplicates: {len(words)}")
    words = delete_duplicates(words)
    print(f"After removing duplicates: {len(words)}")


    out_f.write('\n'.join(words))

