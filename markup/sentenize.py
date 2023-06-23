from collections import Counter
import re
import typing as T

from nltk.tokenize.punkt import (
    PunktLanguageVars,
    PunktTrainer,
    PunktSentenceTokenizer,
)

SakhaPunktLanguageVars = PunktLanguageVars()

POSSIBLE_ABREV = Counter()


class Model:
    action = NotImplemented

    def __init__(self, name):
        self.name = name

    def apply(self, *args, **kwargs) -> T.Any: raise NotImplementedError

    @classmethod
    def load(cls): raise NotImplementedError


class TrainableMixin:
    def train(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class TraningPipeline:
    DEFAULT_PARTS = ("preprocess", "sentenize")

    def __init__(self, trained_model: Model, parts: T.List[str]=None):
        self.model = trained_model

        self.parts = parts = parts or self.DEFAULT_PARTS

        self.pipeline = pipeline = []
        for part in ("preprocess", "sentenize"):
            if part in parts:
                pipeline.append(part)

    def apply(self, data):




class SakhaPunktSentenceTokenizer:
    action = "sentenize"
    ...




def collect_possible_abrev(
    # TODO: the regex is incorrect, need \w\.((\w\.)+PUNC|PUNC)
    #   or this: \b\w\.(?:(\w+\.)+[,:;!?]|[,:;!?])

    textpiece, re_abbrev=re.compile(r'\b(\w{1,4}\.(?:(?:\w{1,4}\.)+)?)(?:[,:;!?]| -)?')
):
    # TODO: use regex here to detect possible abbreviations
    #   to later feed them to tokenizer if it doesn't detect them
    possible_abbrev = re_abbrev.findall(textpiece)
    POSSIBLE_ABREV.update(possible_abbrev)



def train(dataset: Dataset, punkt_trainer: PunktTrainer,
          preprocessor=None, verbose=True):
    for textpiece in tqdm(dataset):
        # TODO: this needs some kind of preprocessing
        if preprocessor:
            textpiece = preprocessor(textpiece)
        # TODO: collect_tok_testdata(textpiece)
        collect_tok_testdata(textpiece)
        punkt_trainer.train(textpiece, finalize=False, verbose=verbose)


