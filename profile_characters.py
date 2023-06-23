import unicodedata
from collections import Counter


def count_chars_notin_alphabet(
    text, alphabet, counter=None, print_text_on_chars=[],
    file_data=None
):
    if counter is None:
        counter = Counter()

    text_printed = False

    for i, ch in enumerate(text):
        if ch not in alphabet:
            counter[ch] += 1
        if ch in print_text_on_chars and not text_printed:
            print(file_data)
            print(f"`{ch}` {unicodedata.name(ch)} {hex(ord(ch))}",
                  text.replace(ch, f'<<{ch}>>'),
                  sep='\n')
            text_printed = True
        elif ch in print_text_on_chars and text_printed:
            print(f"also has `{ch}` {unicodedata.name(ch)} {hex(ord(ch))}")

    return counter


def main(num_paragraphs=20000, translate=False):
    from tqdm import tqdm
    from tidy_string import ALPHABET, make_translation
    from data_models import Dataset, EdersaasJSON

    alphabet = set(ALPHABET)
    # translation = make_translation()
    translation = make_translation(include_alphabet=True, include_letters=True,
                                   map_other_to_none=False)

    dataset = Dataset(return_meta=True)

    counter = Counter()
    # for i, paragraph in enumerate(tqdm(dataset)):
    for i, (paragraph, *metadata) in enumerate(tqdm(dataset)):
        if translate:
            paragraph = paragraph.translate(translation)

        count_chars_notin_alphabet(
            paragraph, alphabet, counter,
            # print_text_on_chars={'\u0473', '\u019f', '\u0275'}
            # print_text_on_chars={'\u2033', '\u04ca', '\u04a3', '\u048b'},
            print_text_on_chars={'\ufe0f', '\u0306'},
            # print_text_on_chars={'\xac'},
            # print_text_on_chars={'\xb8', '\u200d'},
            file_data=metadata
        )
        if i > num_paragraphs:
            break

    for ch, freq in counter.most_common():
        try:
            print(f"`{ch}` {unicodedata.name(ch)} {hex(ord(ch))} : {freq}")
        except ValueError:
            print(f"`{ch}` {'<name N/A>'} {hex(ord(ch))} : {freq}")


if __name__ == "__main__":
    main(100000, True)
