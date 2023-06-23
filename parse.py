from typing import List, Dict, Union, Tuple
import csv
import json
import re
import copy
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag

from lexemes import parse
from utils import HEADERS, write_to_csv, random_delay_adder

# parse = random_delay_adder(1, 1.5)(parse)

import logging
import logging.config

logging.config.fileConfig('logging_parse.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

sakhatyla_site = "https://sakhatyla.ru/"
sakha_link = sakhatyla_site + "translate?q="

words = ("и", "в")

session = requests.session()
session.headers.update(HEADERS)

# def get_similar
#     TODO

Translation = Dict[str, Union[str, Tag]]

PARSED = dict()


def enc_non_alpha(s):
    return ''.join((char.isalpha() and char) or requests.utils.quote(char) for char in s)


def get_word_page(word: str) -> Tuple[str]:
    word_link = "translate?q=" + requests.utils.quote(word)
    link = sakhatyla_site + word_link
    response = session.get(link)

    if response.status_code != 200:
        raise ValueError("Something went wrong while getting result")

    return response.text, link, word_link, word


def save_page(page: str, word: str, folder="sakhatyla.ru"):
    path = f"{folder}/{word}"
    with open(path, 'w', encoding='utf-8') as fout:
        fout.write(page)


def collect_lexical_entries(
        word: str, path: Path = None,
        folder="sakhatyla"
) -> Dict[str, Union[str, List[Translation]]]:
    enc_word = enc_non_alpha(word)
    path = Path(f"{folder}/{enc_word}")
    if not path.exists():
        page, link, _, _ = get_word_page(word)
        save_page(page, enc_word, folder=folder)
        logger.debug(f"saved `{word}` html in {folder}")
    else:
        with open(path, 'r', encoding = 'utf-8') as f:
            page = f.read()
            link = sakha_link + word

    soup = BeautifulSoup(page, 'lxml')

    # достать тэг `<h2>Русский → Якутский</h2>` и смотреть его сестёр дальше
    #   пока не попадём на очередной <h2> (или `<p>ещё переводы</p>`)
    translation_tags: List[Translation] = []
    comment = ''

    # header_soup = soup.find_all('h2', string="Русский → Якутский")
    directions = [
        dict(source_l="ru", targ_l="sa", header="Русский → Якутский"),
        dict(source_l="sa", targ_l="ru", header="Якутский → Русский"),
        dict(source_l="sa", targ_l="en", header="Якутский → Английский")
    ]
    existing_directions = []
    
    for direction_dict in directions:
        direction = direction_dict['header']
        header_soup = soup.find('h2', string=direction)
        if not header_soup:
            logger.debug(f"word: {word}, no `{direction}`")
            continue
        direction_dict.update(dict(header_soup=header_soup))
        existing_directions.append(direction_dict)

    # ru_sa_header = soup.find('h2', string="Русский → Якутский")
    # sa_ru_header = soup.find('h2', string="Якутский → Русский")
    # sa_en_header = soup.find('h2', string="Якутский → Английский")

    # if not header_soup:
    #     comment = f"нет русского перевода. "
    #                # f"есть переводы `{','.join(header_soups)}`")
    #     print(comment)
    #     res = dict(word=word, translations=[], link=link, comment=comment)
    #     return res
    
    for direction_dict in existing_directions:
        header_soup = direction_dict['header_soup']
        
        for tag in header_soup.next_siblings:
            if isinstance(tag, NavigableString):
                continue
            elif tag.name in ('h2', 'p', 'hr'):
                print(f"encountered `{tag.name}`, ending loop (`{tag.string}`)")
                break
            else:
                # TODO: сохранять все тэги или сразу убирать словосочетания?
                # TODO: filtering of words not equal to query should be done here
                source_word_or_phrase = tag.h3.string.strip()
                translation = copy.copy(tag.find('div', class_='article-text'))
    
                # add sentinel value to delimit
                # TODO: may not be needed with copies?
    
                lexical_category_tag = tag.find('div', class_='article-category')
                lexical_category = ''
                if lexical_category_tag:
                    lexical_category = lexical_category_tag.string.split(': ')[1]

                entry_dict = {k: v for k, v in direction_dict.items()
                              if k not in ('header', 'header_soup')}
                entry_dict.update(dict(source=source_word_or_phrase,
                    translation=translation, lexical_category=lexical_category))
                translation_tags.append(entry_dict)
    
    logger.debug(f'Successfully parsed `{sakha_link+word}`')
    res = dict(word=word, translations=translation_tags, link=link, comment=comment)
    
    return res


def parse_word_results(word: str, results: List[Dict[str, str]]):
    word_res = collect_lexical_entries(word)

    for i, entry in enumerate(word_res['translations']):
        # TODO: check if entry['source'] == word ?
        entry_title = entry['source']
        logger.debug(f"word `{word}` entry {i} ({entry_title}),"
                     f"link: {sakha_link + requests.utils.quote(word)}")
        if entry_title in PARSED:
            logger.info(f"skipping {entry_title} (already parsed)")
            continue
        if entry['source_l'] == 'sa' and entry['targ_l'] == 'ru':
            if entry['source'][-1] == "=":
                entry['eq_pos'] = "V"

            res_d = {}
            res_d.update(entry)

            try:
                res_d['translation'] = parse(entry['translation'], prettify=True)
            except (ValueError, AttributeError, TypeError) as e:
                logger.error(e)
                res_d['error'] = e

            results.append(res_d)

            logger.debug(f"result is {res_d}")

        PARSED[entry_title] = None
        with open("res.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    # filename = "collect_words/words.txt"
    filename = "words_full.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')

    results = []
    met_start = False
    for word in words:
        parse_word_results(word, results)
