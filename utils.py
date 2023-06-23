import random
from time import sleep

from pathlib import Path

HEADERS = {
    # 'authority': 'www.kith.com',
    'cache-control': 'max-age=0',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
    'sec-fetch-dest': 'document',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'accept-language': 'ru;q=0.9, en-US;q=0.8, en;q=0.7',
}

DATA_FOLDER = Path("./data-metadata")


def random_delay_adder(min, max):
    def add_random_delay(func):
        def delayed_func(*args, **kwargs):
            sleep(random.uniform(min, max))

            return func(*args, **kwargs)

        return delayed_func
    return add_random_delay


def write_to_csv(entries, filename='translations_{}.csv'):
    if not Path(filename.format('')).exists:
        filename = filename.format('')
    else:
        maxind=-1
        for file in sorted(Path('.').glob('translations_?*.csv')):
            ind = int(file.stem.lstrip('translations_'))
            if ind > maxind:
                maxind = ind
        if maxind == -1:
            maxind = 0
        filename = filename.format(maxind+1)

    with open(filename, 'w', encoding='utf-8', newline='') as csvout:
        # TODO: add field for index of (possibly multirow) article on the page)
        fieldnames = ['word', 'rus', 'sense', 'translation', 'example',
                      'lexical_category', 'comment', 'link']
        writer = csv.DictWriter(csvout, fieldnames=fieldnames)

        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def _get_rus_alphabet(file=DATA_FOLDER / Path("ru_alphabet.txt")):
    with open(file, 'r', encoding='utf-8') as f:
        letters = frozenset(f.read().split())
    return letters


def _get_sakha_alphabet(file=DATA_FOLDER/Path("sa_alphabet.txt"), res_container="frozenset",
                        lower_only=False):
    with open(file, 'r', encoding='utf-8') as f:
        letters = f.read().split()
    if lower_only:
        letters = letters[1::2]
    if res_container == "frozenset":
        return frozenset(letters)
    return letters


def _get_sakhaonly_letters(file=DATA_FOLDER / Path("sa_uniquealphabet.txt"), res_container="frozenset"):
    with open(file, 'r', encoding='utf-8') as f:
        letters = f.read().split()
    if res_container == "frozenset":
        return frozenset(letters)
    return letters


SAKHA_ALPHABET = _get_sakha_alphabet()
SAKHAONLY_LETTERS = _get_sakhaonly_letters()
RUS_ALPHABET = _get_rus_alphabet()
# there are no RUSONLY_LETTERS, Sakha alphabet uses whole Russian alphabet + 5 unique