from typing import Dict, List, Tuple
import re

RU_LC = 'абвгдеёжзийклмнопрстуфхцчшщъьыэюя'
# RU_LC = "".join([chr(_i) for _i in range(ord("а"), ord("я"))] + ["ё"])
RU_UC = 'АБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'
SA_ONLY_LC = 'өүҥһҕ'
SA_ONLY_UC = 'ӨҮҤҺҔ'
LAT_LC = 'abcdefghijklmnopqrstuvwxyz'
LAT_UC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
PUNC = r""",.:;!?'"(){}[]<>-=+_/\|%@#$&№*^"""  # TODO: what to do with `_`?
WHITESPACE = " \t\n\r"
# these are unicode characters potentially useful for tokenization
PUNC_RARE_USEFUL = [
    '−',  # minus sign (\u2212)
    '°'  # degree sign (\xb0)
]

# ₳฿₿￠₡¢₢₵₫€￡£₤₣ƒ₲₭Ł₥₦₽₱＄$₮ℳ₶₩￦¥￥₴₸¤₰៛₪₯₠₧﷼円元圓㍐원৳₹₨৲௹
# TODO: translation may be needed from unusual signs (wide dollar / pound) to normal
CURRENCIES = """\u20B3\u0E3F\u20BF\uFFE0\u20A1\u00A2\u20A2\u20B5\u20AB\u20AC\uFFE1
\u00A3\u20A4\u20A3\u0192\u20B2\u20AD\u0141\u20A5\u20A6\u20BD\u20B1\uFF04\u0024\u20AE
\u2133\u20B6\u20A9\uFFE6\u00A5\uFFE5\u20B4\u20B8\u00A4\u20B0\u17DB\u20AA\u20AF\u20A0
\u20A7\uFDFC\u5186\u5143\u5713\u3350\uC6D0\u09F3\u20B9\u20A8\u09F2"""

ALPHABET = list(
    RU_LC + RU_UC + SA_ONLY_LC + SA_ONLY_UC + LAT_LC + LAT_UC
    + DIGITS + PUNC + WHITESPACE
)

ALPHABET_EXT = ALPHABET + list(CURRENCIES)

# this can perhaps be multilingual
symbols = {
    '\xad': '',  # soft hyphens
    '\u2012': '-', '\u2013': '-',   # figure dash, en dash
    # '\u2212': '-',                # minus sign
    '\u200b': '', '\u200c': '',     # zero-width space and non-joiner
    # '\u200d': '',                 # zero-width joiner
    # '\u2800': '',                 # braille pattern blank (IG new line)
    '\u2014': ' -- ',               # em dash TODO: is it actually needed?
    '\xa0': ' ', '\u202f': ' ',     # non-breaking space (and narrow)
    '\u2018': '"', '\u2019': '"',   # single quotes, left and right `‘`, `’`
    '\xab': '"', '\xbb': '"',       # Double Angle Quotation Marks `«`, `»`
    '\u201c': '"', '\u201d': '"',   # Double Quotation Marks `“`, `”`
    # TODO: there must be more quotes
    '\u201e': '"',                  # double low-9 quote `„`
    # '\u2033': '"',                # Double Prime `″`
    '\xd7': '*',                    # multiplication sign `×`
}

# TODO: potential conversion to make - number symbols to literal numbers
#  https://unicode-table.com/en/sets/numerals/
#  https://unicode-table.com/en/sets/superscript-and-subscript-numbers/
#  https://unicode-table.com/en/sets/roman-numerals/ (roman numbers?)

# TODO: что с эмодзи? потенциально их есть смысл оставить для продвинутого нлп
#  типа intent detection. Текстов якутских много, много интернетных, почему нет?
#  у них часто есть \ufe0f

MISC_RANGE = '\u2600-\u26FF'                    # Miscellaneous Symbols
DINGBATS_RANGE = '\u2700-\u27BF'                # Dingbats
MISC_SYMPIC_RANGE = '\U0001f300-\U0001f5ff'     # Miscellaneous Symbols and Pictographs
EMOJI_RANGE = '\U0001F600-\U0001F64F'           # Emoticons (Emoji)
TRANSP_MAP_RANGE = '\U0001F680-\U0001F6FF'      # Transport and Map Symbols
SUPP_SYMPIC_RANGE = '\U0001F900-\U0001F9FF'     # Supplemental Symbols and Pictographs
SYMPIC_EXTA_RANGE = '\U0001FA70-\U0001FAFF'     # Symbols and Pictographs Extended-A (not all are used?)
SKIN_COLORS_RANGE = '\U0001f3fb-\U0001f3ff'     # 5 skin color modifiers
ranges = [f"{range}" for range in (MISC_RANGE, DINGBATS_RANGE, MISC_SYMPIC_RANGE,
                                   EMOJI_RANGE, TRANSP_MAP_RANGE, SUPP_SYMPIC_RANGE,
                                   SYMPIC_EXTA_RANGE)]

ranges_str = ''.join(ranges)
regex_one = re.compile(f"[{ranges_str}]")

# NOTE: https://www.unicode.org/Public/emoji/14.0/emoji-zwj-sequences.txt
EMOJI_RE = re.compile(
    f"("
    f"[\U0001F100-\U0001F1FF][\U0001F100-\U0001F1FF]"               # flags from two reg symbols
    f"| "
    f"[{ranges_str}]"                                               # base emoji
        f"[{SKIN_COLORS_RANGE}]?"                                   # skin color
        f"(?:\u200d[{ranges_str}]\ufe0f?[{SKIN_COLORS_RANGE}]?)*"   # combinations with other emojis   
    f")",
)


def is_emoji(string, regex=EMOJI_RE):
    return bool(regex.fullmatch(string))


# alternative. Source: https://stackoverflow.com/a/69866962
_alternative_EMOJI_RE_str = '''(
    (\ud83c[\udde6-\uddff]){2}
    |
    ([#*0-9]\u20e3)
    |
    (\u00a9|\u00ae|[\u2000-\u3300]|[\ud83c-\ud83e][\ud000-\udfff])
        (
            (\ud83c[\udffb-\udfff])?
            (\ud83e[\uddb0-\uddb3])?
            (\ufe0f?\u200d
                ([\u2000-\u3300]|[\ud83c-\ud83e][\ud000-\udfff])\ufe0f?
            )?
        )*
)'''


letters = {
    '\u019f': '\u04e8', '\u0275': '\u04e9',  # latin o bar to cyrillic o-bar
    '\u0472': '\u04e8', '\u0473': '\u04e9',  # cyrillic fita to cyrillic o-bar
    '\u048b': 'й',  # I with tail `ҋ` to common `й`
    '\u04A2': 'ҥ'
}


class Replacement:
    GROUP_NAME_RE = re.compile('\?P<(\w+)>')

    def __init__(self, **keyvals):
        self.keys_dict = {}
        self.repl_dict = {}

        for key, val in keyvals.items():
            self.keys_dict[key] = None
            mo = self.GROUP_NAME_RE.search(key)
            if mo:
                self.repl_dict[mo.group(1)] = val
            else:
                self.repl_dict[key] = val

    def keys(self):
        return self.keys_dict.keys()

    def get(self, *args, **kwargs):
        return self.repl_dict.get(*args, **kwargs)

    def __str__(self):
        return f"<{self.keys_dict.__repr__()}, {self.repl_dict.__repr__()}>"

# TODO: many-to-X
# и + \u0306 (и + combining breve) -> й
# many_to_one_letters = {
#     r'(?P<spaces>\s\s+)': None,
#     'spaces': '\x20',
#     '(?P<repeat>.)\\1{2}\\1+': None,
#     'repeat': r'\g<repeat>'*3,  # TODO: these shouldn't be used when saving text
#     'и\u0306': 'й',
#     'арга': 'INF',  # TEST
# }


many_to_one_re = {
    r'(?P<repeat>.)\1{3,}': r'\g<repeat>' * 3,
    r'(?P<spaces>\s\s+)': '\x20',
}
many_to_one_letters = {
    'и\u0306': 'й',
    # 'арга': 'INF',  # TEST
}
many_to_one = (many_to_one_re, many_to_one_letters)


def describe(string: str):
    for ch in string:
        print(f"{ch} {hex(ord(ch))}")


class MappingWithDefault(dict):
    def __missing__(self, key):
        return None


def check_pairwise_prefixhood(l: List[str]):
    prefixes = {}
    sorted_l = sorted(l)
    for i, possible_prefix in enumerate(sorted_l):
        # print(f"pref is {possible_prefix}")
        for j, word in enumerate(sorted_l[i + 1:]):
            # print(f"word is {word}")
            if possible_prefix[0] != word[0]:
                # print(f"quitting word, checking next prefix")
                break
            if word != possible_prefix and word.startswith(possible_prefix):
                # print(f"found prefix")
                prefixes.setdefault(possible_prefix, []).append(word)
    return prefixes


def multiple_replace_text(
        replacement_dict: dict, text: str
) -> Tuple[str, re.Pattern[str]]:

    # Create a regular expression  from the dictionary keys
    # keys = sorted(dict.keys(), key=len, reverse=True) # TODO: reverse?
    keys = replacement_dict.keys()
    prefixes = check_pairwise_prefixhood(keys)
    if prefixes:
        print(f"dictionary contains prefixed keys:\n{prefixes}")
        keys = sorted(keys)
        print(f"default policy is alphabetic sorting:\n{keys}"
              f"\nthis gives priority to strict prefixes, but not necessarily to shortest match")

    print(keys)
    # regex = re.compile(rf"{'|'.join(map(re.escape, keys))}")
    regex = re.compile(rf"{'|'.join(keys)}")

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacement_dict.get(mo.group(0), mo.group(0)), text), regex


def multiple_replace_regex(
    replacement_dict: dict, text: str
) -> Tuple[str, re.Pattern[str]]:
    for templ, repl in replacement_dict.items():
        text = re.sub(templ, repl, text)

    return text


def multiple_replace(re_dict: Dict[str, str], letters_dict: Dict[str, str], text: str) -> str:
    text = multiple_replace_regex(re_dict, text)
    text, _ = multiple_replace_text(letters_dict, text)
    return text


def make_translation(include_alphabet=True, include_letters=True,
                     extra_translations=None, map_other_to_none=False):
    translation = symbols
    if include_alphabet:
        translation.update({ch: ch for ch in ALPHABET})
    if include_letters:
        translation.update(letters)
    if extra_translations:
        translation.update(extra_translations)
    translation = str.maketrans(translation)
    if map_other_to_none:
        translation = MappingWithDefault(translation)

    return translation


# TODO: tokenization may require replacements like `(\w+)(PUNC)(\w+)` -> `\1 \2 \3`


def make_string_tidier(translation, add_space=False, enspace_puncs=[',']):
    if add_space:
        import re
        pat = re.compile(rf"(\w+)([{''.join(enspace_puncs)}])(\w+)")

        def tidy_string(text):
            text = pat.sub("\1\2 \3", text)
            return text.translate(translation)

    else:
        def tidy_string(text):
            return text.translate(translation)

    return tidy_string


class Tidier:
    """Tidies the string"""
    def __init__(self, translation: Dict = None, multi_map: Dict = None,
                 multi_map_func: [[Dict, str], str] = None):
        if not translation:
            translation = make_translation()
        self.translation = translation

        if not multi_map:
            multi_map = many_to_one
        self.multi_map = multi_map

        if not multi_map_func:
            multi_map_func = multiple_replace
        self.multi_map_func = multi_map_func

    def __call__(self, text: str):
        return self.multi_map_func(*self.multi_map, text).translate(self.translation)


print(ALPHABET)
