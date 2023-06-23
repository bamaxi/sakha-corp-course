import re
from collections import deque
import random
from time import sleep

from typing import Any, MutableSequence, List, Dict, Union, Tuple

import requests

from utils import HEADERS, _get_sakha_alphabet

import logging
import logging.config

logging.config.fileConfig('logging_collect_words.conf')
logger = logging.getLogger(__name__)

HEADERS['accept'] = "application/json,text/javascript"

# TODO: excluding letters used strictly in russian loans is an option,
#  should it be done?
SAKHA_ALPHABET = _get_sakha_alphabet(lower_only=True, res_container="list")
LEN_SAKHA_ALPHABET = len(SAKHA_ALPHABET)

VOWELS=r'[аоиуыөүеёэюя]'

sakhatyla_site = "https://sakhatyla.ru/"
sakha_suggest_link = sakhatyla_site + "api/articles/suggest?query="

session = requests.Session()
session.headers.update(HEADERS)

# this is what the server outputs, less than that means we found all with this prefix
NUM_MAX_RES = 10

prefixes_to_exhaust = deque()
prefixes_next_letter = deque()


def append_not_none(collection: MutableSequence, value: Any) -> None:
    if value is not None:
        collection.append(value)


def random_delay_adder(min, max):
    def add_random_delay(func):
        def delayed_func(*args, **kwargs):
            sleep(random.uniform(min, max))

            return func(*args, **kwargs)

        return delayed_func
    return add_random_delay


@random_delay_adder(1.5, 2)
def get_prefix_json(prefix: str) -> Tuple[List[Dict[str, Union[str, int]]], str]:
    link = sakha_suggest_link + requests.utils.quote(prefix)
    response = session.get(link, headers=HEADERS)

    if response.status_code != 200:
        raise ValueError("Something went wrong while getting result")

    logger.debug(f"in `get_prefix_json`, prefix is `{prefix}`,"
                 # f"res is:\n{response.text}")
                 f"res len is {len(response.json())}")
    return response.json(), link


def get_next_letter_prefix(
        prefix,
        three_vowels=re.compile(rf'{VOWELS}{3}$', re.M),
        # signs_after_vowel=re.compile(rf'')
):
    # TODO: understand if it captures everything and how it works with prefixes
    #   that end in "...(н|д)..."

    if prefix[-2] in ('д', 'н') and prefix[-1] == 'ь':
        last_char = prefix[-2] + 'ь'
        prefix_part_kept = prefix[:-2]
    else:
        last_char = prefix[-1]
        prefix_part_kept = prefix[:-1]

    # logger.debug(f"prefix is {prefix} (len {len(prefix)}), last char is {last_char} (len {len(last_char)})"
    #              f"last char in alph? {last_char in SAKHA_ALPHABET}. last_char index: ")

    next_letter_i = SAKHA_ALPHABET.index(last_char)+1
    if next_letter_i >= LEN_SAKHA_ALPHABET:
        return None

    proposed_prefix = prefix_part_kept + SAKHA_ALPHABET[next_letter_i]

    # heuristics for skipping bad prefixes go here
    # if proposed_prefix[-2] in ('д', 'н') and proposed_prefix[-1] == 'ь'
    msg = None
    if three_vowels.search(proposed_prefix[-3:]):
        msg = f"skipping {proposed_prefix} (3V)"

    if msg:
        logger.debug(msg)
        return get_next_letter_prefix(proposed_prefix)

    return proposed_prefix


def get_new_first_letter_prefix(prefix):
    next_letter_i = SAKHA_ALPHABET.index(prefix[1])+1
    if next_letter_i >= LEN_SAKHA_ALPHABET:
        return None
    # it appears search is only performed with len >= 2 prefixes
    return SAKHA_ALPHABET[next_letter_i] + "а"


# TODO: functions seem to differ only in the array used
def exhaust_prefix():
    prefix = prefixes_to_exhaust.pop()
    items_list, link = get_prefix_json(prefix)

    # no matter whether there are results
    # prefix one letter up from current must be checked
    append_not_none(prefixes_next_letter, get_next_letter_prefix(prefix))
    # prefixes_next_letter.append(get_next_letter_prefix(prefix))

    if not items_list:
        return None

    words = [item["Title"] for item in items_list]
    last_word = words[-1]

    # if all words with current prefix aren't exhausted by this search
    if len(words) == NUM_MAX_RES:
        next_prefix = last_word[:len(prefix) + 1]
        prefixes_to_exhaust.append(next_prefix)
        # append_not_none(prefixes_next_letter, get_next_letter_prefix(next_prefix))
        # prefixes_next_letter.append(get_next_letter_prefix(next_prefix))

    return words


def continue_alphabet():
    prefix = prefixes_next_letter.pop()
    items_list, link = get_prefix_json(prefix)

    # no matter whether there are results
    # and if there are whether there is longer prefix to exhaust (below),
    # we check prefix with last letter changed to the next in alphabet
    append_not_none(prefixes_next_letter, get_next_letter_prefix(prefix))
    # prefixes_next_letter.append(get_next_letter_prefix(prefix))

    if not items_list:
        return None

    words = [item["Title"] for item in items_list]
    last_word = words[-1]

    # there are suggestions for prefix
    # if there are 10 results - the max - prefix should increase to last number
    if len(words) == NUM_MAX_RES:
        next_prefix = last_word[:len(prefix) + 1]
        prefixes_to_exhaust.append(next_prefix)

    return words


def enlist_words(out_filename="words.txt"):
    """
    get (hopefully all) words from the site, that could be suggested in search line

    two queues are used: FIRST, `...exhaust` keeps track of prefixes that could
    be expanded     by further consuming characters from last suggest results.
    Once the number of suggest results is less than `NUM_MAX_RES` (10),
    the prefix that started the expansion is considered exhausted.

    Indepently from (and during each step of) the above we move the last letter
    of the current suffix (some heuristics allow to skip some combinations) down
    the alphabet and add it to the SECOND queue `prefixes_next_letter`

    The `exhaust_prefix()` function
    :return:
    """
    # TODO: unclear how the jump is made from `l...` to `(l+1)...`
    # TODO: if prefix is e.g. `аб` and first res is `аба` we can change prefix for first res
    #   then for exhaust() we can try first_res + 1 symb from last res
    #   (but first res must be last res's prefix!!!)

    while prefixes_next_letter:
        previous_prefix = prefixes_next_letter[-1] if prefixes_next_letter else None
        logger.debug(f"before `continue`, deques are `{prefixes_to_exhaust}`, `{prefixes_next_letter}`")
        words = continue_alphabet()
        logger.debug(f"words are: {words}, deques are `{prefixes_to_exhaust}`, `{prefixes_next_letter}`")
        if words:
            res.update(dict.fromkeys(words))

        while prefixes_to_exhaust:
            logger.debug(f"before `exhaust`, deques are `{prefixes_to_exhaust}`, `{prefixes_next_letter}`")
            words = exhaust_prefix()
            logger.debug(f"after `exhaust`, words are: {words},"
                         f"deques are `{prefixes_to_exhaust}`, `{prefixes_next_letter}`")
            if words:
                res.update(dict.fromkeys(words))

        # test code
        # logger.warning(f"manually deleting everything from queues to check next letter jumping")
        # prefixes_to_exhaust.clear()
        # prefixes_next_letter.clear()

        if not prefixes_next_letter and previous_prefix:
            append_not_none(prefixes_next_letter, get_new_first_letter_prefix(previous_prefix))

        # logger.info(f"res after {previous_prefix}:\n{res.keys()}")
        with open(out_filename, 'w', encoding='utf-8') as outf:
            outf.write('\n'.join(res.keys()))


if __name__ == "__main__":
    res = {}
    prefixes_next_letter.append('аа')  # a place to start
    logger.info(SAKHA_ALPHABET)

    enlist_words()
