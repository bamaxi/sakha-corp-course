import re
from collections import Counter
from pathlib import Path
import pickle

from tqdm import tqdm
from nltk.tokenize.punkt import PunktTrainer

from data_models import Dataset, EdersaasJSON
from tidy_string import make_translation


POSSIBLE_ABREV = Counter()


def collect_tok_testdata(
    # TODO: the regex is incorrect, need \w\.((\w\.)+PUNC|PUNC)
    textpiece, re_abbrev=re.compile(r'\b(\w{1,4}\.(?:(?:\w{1,4}\.)+)?)(?:[,:;!?]| -)?')
):
    # TODO: use regex here to detect possible abbreviations
    #   to later feed them to tokenizer if it doesn't detect them
    possible_abbrev = re_abbrev.findall(textpiece)
    POSSIBLE_ABREV.update(Counter(possible_abbrev))


def train(dataset: Dataset, punkt_trainer: PunktTrainer,
          preprocessor=None, verbose=True):
    for textpiece in tqdm(dataset):
        # TODO: this needs some kind of preprocessing
        if preprocessor:
            textpiece = preprocessor(textpiece)
        # TODO: collect_tok_testdata(textpiece)
        collect_tok_testdata(textpiece)
        punkt_trainer.train(textpiece, finalize=False, verbose=verbose)


def main():
    save_dir = './punkt_models'
    tokenizer_filename = 'sakha_edersaas.pickle'

    dataset = Dataset()
    punkt_trainer = PunktTrainer()

    translation = make_translation(include_letters=True)
    preprocessor = lambda str: str.translate(translation)

    train(dataset, punkt_trainer, preprocessor)

    # TODO: needed not here but perhaps before usage?
    punkt_trainer.finalize_training()
    params = punkt_trainer.get_params()

    with open(Path(save_dir) / Path(tokenizer_filename), 'wb') as f:
        pickle.dump(params, f)

    print(f'saved tokenizer trainer at {tokenizer_filename}')

    print(f"tokenizer abbreviations:\n{punkt_trainer.get_params().abbrev_types}")
    print(f"tokenizer collocations:\n{punkt_trainer.get_params().collocations}")
    print(f"tokenizer sentence starters:\n{punkt_trainer.get_params().sent_starters}")
    # print(f"tokenizer ortho_context:\n{punkt_trainer.get_params().ortho_context}")

    with open('possible_abbrev', 'w', encoding='utf-8') as f:
        f.write(f"abbr\tcount\tis_included\n")
        for abbr, count in POSSIBLE_ABREV.most_common():
            included = True
            if abbr.strip('.') not in params.abbrev_types:
                included = False
                print(f"possible abbr `{abbr}` missed")
            f.write(f"{abbr}\t{count}\t{int(included)}\n")


if __name__ == "__main__":
    main()
