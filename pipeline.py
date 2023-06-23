import typing as T
import pickle
import re
import warnings

from tqdm import tqdm
from nltk.tokenize.punkt import (
    PunktSentenceTokenizer,
    PunktParameters
)
from spacy import load as spacy_load
from spacy.language import Tokenizer
from spacy.tokens import Token

from tidy_string import make_translation, EMOJI_RE, is_emoji, describe, Tidier
from data_models import Dataset
from markup.foma_utils import (
    analyze_form_class
)

# from LangAnalysis.lang_analysis import Language_Determiner

sent_tokenizer_params_pickled = 'sakha_edersaas_0.pickle'


import logging
logging.basicConfig(filename='pipeline.log')
logger = logging.getLogger()


def unpickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


InputType = T.TypeVar
OutputType = T.TypeVar



class PipelineComponent:
    __name__ = "component"
    def apply(self, input: InputType) -> OutputType: ...


class BaseSentenizer(PipelineComponent):
    __name__ = "sentenizer"


class PunktSentenizer(BaseSentenizer):
    def __init__(self, punkt_sentenizer: PunktSentenceTokenizer):
        self.model = punkt_sentenizer

    def apply(self, input: str) -> T.List[str]:
        return self.model.tokenize(input)

    @classmethod
    def from_pickled_params(cls, filename: str, verbose=False):
        warnings.warn("unpickling is unsafe")
        _params: PunktParameters = unpickle(filename)
        params = _params.get_params()

        model = PunktSentenceTokenizer(train_text=params, verbose=verbose)
        return cls(model)


class MorphAnalyzer(PipelineComponent):
    __name__ = "morph"


class FomaAnalyzer(MorphAnalyzer):
    def __init__(self, foma_model):
        self.model = foma_model

    def apply(self, input: str) -> str:
        return NotImplemented


class Pipeline:
    def __init__(
        self, sent_tokenizer_params, tidying_translation=None,
        tokenizer: Tokenizer = None, preprocessor: Tidier = None
    ):
        if not tidying_translation:
            tidying_translation = make_translation()
        self.translation = tidying_translation
        # self.preprocessor = lambda text: text.translate(tidying_translation)
        if not preprocessor:
            preprocessor = Tidier()
        self.preprocessor = preprocessor

        self.sent_tokenizer_params = sent_tokenizer_params
        self.sent_tokenizer = PunktSentenceTokenizer(train_text=sent_tokenizer_params)
        if not tokenizer:
            emoji_getter = lambda token: is_emoji(token.text)
            Token.set_extension("is_emoji", getter=emoji_getter)
            nlp = spacy_load("ru_core_news_lg")
            tknzr = nlp.tokenizer
            # fix regex to allow full emoji matching
            # TODO: further fixes? does
            prefix_flags = tknzr.prefix_search.__self__.flags
            tknzr.prefix_search = re.compile(
                f"{EMOJI_RE.pattern}|{tknzr.prefix_search.__self__.pattern}",
                flags=prefix_flags).search

            suffix_flags = tknzr.suffix_search.__self__.flags
            tknzr.suffix_search = re.compile(
                f"{EMOJI_RE.pattern}|{tknzr.suffix_search.__self__.pattern}",
                flags=suffix_flags).search

            # TODO: compound emoji as infix is still problematic for some reason
            #  (.find_infix() detects it but then something happens)
            infix_flags = tknzr.infix_finditer.__self__.flags
            tknzr.infix_finditer = re.compile(
                f"{EMOJI_RE.pattern}$|{tknzr.infix_finditer.__self__.pattern}",
                flags=infix_flags).finditer

            tokenizer = tknzr
            del nlp

        self.tokenizer = tokenizer

        # TODO: this should return result similar to russian model
        # self.sakha_analyzer = fst.apply_up
        self.sakha_analyzer = analyze_form_class

    def apply_sent(self, sent):
        # token_func = self.apply_token

        tokens = self.tokenizer(sent)
        result = []
        for token in tokens:
            # print(f"token: {token}")
            # res = self.apply_token(token)
            text = token.text
            if token.is_alpha:
                res = self.sakha_analyzer(text)
                if res is None:
                    res = {'pos': "UNK", "value": text}
            elif token.is_punct:
                res = {'pos': 'PUNC', 'value': text}
            elif token.is_digit or token.like_num:
                res = {'pos': 'NUMRL', 'value': text}
            elif token._.is_emoji:
                res = {'pos': 'EMJ', 'value': text}
            elif token.like_url:
                res = {'pos': 'URL', 'value': text}
            elif token.is_space:
                print(describe(text))
                continue
            else:
                print(describe(text))
                res = {'pos': '?', 'value': text}
                # continue
            result.append(res)
            res = {}
            logger.warning(f"{result}")
        return result

    def apply_text(self, text):
        # tidier_text = text.translate(self.translation)
        tidier_text = self.preprocessor(text)
        sentences = self.sent_tokenizer.tokenize(tidier_text)

        result = []
        for sentence in sentences:
            print(f"sent: {sentence}")
            res = self.apply_sent(sentence)
            result.append(res)
        return result

    def operation(self, chunk):
        print(f"{chunk} inside concurrent insert")
        print(f"do some operations with {chunk}")
        return chunk**2

    def process_text(self, text_iterable):
        results = []
        # results_append = results.append
        # with ProcessPoolExecutor(max_workers=6) as executor:
        #     for result in executor.map(self.apply_text, text_iterable):
            # for result in executor.map(self.operation, range(20)):
        for text in text_iterable:
            result = self.apply_text(text)
            print(result)
            results.append(result)
        return results


def main(n_iter=50):
    sent_tokenizer_params = unpickle(sent_tokenizer_params_pickled).get_params()

    pipeline = Pipeline(sent_tokenizer_params)

    # LD = Language_Determiner()
    # LD.load_categories()

    dataset = Dataset(max_iter=n_iter)
    print(sent_tokenizer_params)

    iter = tqdm(dataset)
    iter.set_description_str(f"items (paragraphs - main)")
    paragraphs_iter = iter
    # paragraphs_iter = (paragraph for i, paragraph in enumerate(iter) if i < n_iter)
    results = pipeline.process_text(paragraphs_iter)

    # for paragraph in paragraphs_iter:
    #     print(paragraph)
    # print(results)
    # results = []


    # with ProcessPoolExecutor(max_workers=6) as executor:
    #     # for result in executor.map(pipeline.apply_text, paragraphs_iter):
    #     #     results.append(result)
    #     for result in executor.map(pipeline.operation, range(20)):
    #         results.append(result)

            # langs = LD.compare_by_rank(paragraph)
            # print(langs)
            # print(paragraph)

            # result = pipeline.apply_text(paragraph)
            # print(result)
            # results.append(result)
    # return results


if __name__ == "__main__":
    # import cProfile, pstats, io
    # from pstats import SortKey
    #
    # pr = cProfile.Profile()
    # pr.enable()

    main(n_iter=5)

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(0.5)
