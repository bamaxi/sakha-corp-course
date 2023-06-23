from typing import Literal, Union, List, Tuple
import random

from pathlib import Path
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ModuleNotFoundError:
    TQDM_AVAILABLE = False


class Data:
    def do_sampling(self):
        files = self.files
        if self.sample_ratio:
            self.files = random.sample(
                files, int(len(files) * self.sample_ratio)
            )
        elif self.sample_n:
            self.files = random.sample(files, self.sample_n)
        else:
            self.files = files

    TEXT_RETURNING_FUNCTIONS = [
        # 'get_article', 'get_paragraph',
        '__getitem__', '__iter__'
    ]

    def __setattr__(self, key, value):
        # print(f"in Data.__setattr__, args: `{key}` `{value if key != 'files' else '%FILES%'}`")
        if key == "return_meta":
            text_returning_functions = self.__class__.TEXT_RETURNING_FUNCTIONS
            cls = self.__class__
            if value is False:
                new_cls = type(f"_{cls.__name__}_NOMETA", (cls,), {})
                for func_name in text_returning_functions:
                    setattr(new_cls, func_name, getattr(cls, f"_{cls.__name__}{func_name}_nometa"))
                self.__class__ = new_cls
            elif value is True:
                new_cls = type(f"_{cls.__name__}_META", (cls,), {})
                for func_name in text_returning_functions:
                    setattr(new_cls, func_name, getattr(cls, f"_{cls.__name__}{func_name}_meta"))
                self.__class__ = new_cls
            else:
                raise ValueError(f"incorrect parameter specification: `return_meta`=`{value}`")

        if key == "iter_what" and value == 'paragraphs':
            super().__setattr__("_iter_text_piece", getattr(self, "_iter_paragraphs"))
        elif key == "iter_what" and value == 'articles':
            super().__setattr__("_iter_text_piece", getattr(self, "_iter_articles"))
        elif key == "iter_what":
            raise ValueError(f"incorrect parameter specification: `iter_what`=`{value}`")

        super().__setattr__(key, value)


class EdersaasJSON(Data):
    NAME = 'edersaas'

    def __init__(self, folder, file_extension='json',
                 sample_ratio=None, sample_n=None,
                 return_meta=False, paragraph_joiner='\n\n',
                 iter_what: Literal['paragraphs', 'articles'] = 'paragraphs',
                 progress_bar=False):
        self.folder = folder
        self.file_extension = file_extension
        files = list(Path(folder).glob(f'*.{file_extension}'))
        self.files = files
        if self.file_extension != 'json':
            raise NotImplementedError(f"Parsing not implemented for non-json files")

        self.sample_ratio = sample_ratio
        self.sample_n = sample_n
        self.do_sampling()

        self.paragraph_joiner = paragraph_joiner

        self.progress_bar = progress_bar

        # TODO: if we can't support changing mechanisms after instantiation and
        #   halfway through iterating, these could be initialized in init
        #   instead of parent __setattr__
        #   however current implementation allows inheritance
        # this creates duplicate get_ methods and __iter__ from existing meta/nometa methods
        #   (see Data)
        self.return_meta = return_meta
        # this creates _iter_text_piece
        self.iter_what = iter_what

    # TODO: may need to switch to JSONDecoder
    def parse_json_file(self, file=None, filename=None):
        if filename:
            path = Path(self.folder) / Path(filename)
        elif file:
            path = file
        with open(path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        return file_data

    def __getitem___nometa(self, item: Union[str, Tuple[Union[str, int]]]):
        if isinstance(item, (tuple, list)):
            articles = self.parse_json_file(filename=item[0])
            article = articles[item[1]]
            paragraphs = article['text']
            if len(item) == 2:
                # we have (file, article_index)
                return self.paragraph_joiner.join(paragraphs)
                # return self.get_article(*item)
            elif len(item) == 3:
                # we have (file, article_index, paragraph_index)
                return paragraphs[item[2]]
                # return self.get_paragraph(*item)
        elif isinstance(item, str):
            return self.parse_json_file(item)
        raise ValueError(f"unknown index type: `{item}`")

    def __getitem___meta(self, item: Union[str, Tuple[Union[str, int]]]):
        print(f"item is {item}, type {type(item)}")
        if isinstance(item, (tuple, list)):
            articles = self.parse_json_file(filename=item[0])
            article = articles[item[1]]
            paragraphs = article['text']
            meta = dict(source=self.NAME, index=item,
                        link=article['link'], date=article['date'])
            if len(item) == 2:
                # we have (file, article_index)
                return self.paragraph_joiner.join(paragraphs), meta
            elif len(item) == 3:
                # we have (file, article_index, paragraph_index)
                return paragraphs[item[2]], meta
        elif isinstance(item, str):
            return self.parse_json_file(filename=item)
        raise ValueError(f"unknown index type: `{item}`")

    def _iter_paragraphs(self, article):
        yield from article['text']

    def _iter_articles(self, article):
        yield self.paragraph_joiner.join(article['text'])

    def __iter___meta(self):
        if self.progress_bar:
            files_iter = tqdm(self.files)
            files_iter.set_description(f"items (Dataset)")
            files_iter = enumerate(files_iter)
        else:
            files_iter = self.files
        # files_iter = self.files if not self.progress_bar else tqdm(self.files)
        for file in files_iter:
            file_data = self.parse_json_file(file=file)
            for i, article in enumerate(file_data):
                meta = dict(source=self.NAME, index=(file, i),
                            link=article['link'], date=article['date'])
                yield from ((text_piece, meta)
                            for text_piece in self._iter_text_piece(article))

    def __iter___nometa(self):
        if self.progress_bar:
            files_iter = tqdm(self.files)
            files_iter.set_description(f"items (Dataset)")
            files_iter = enumerate(files_iter)
        else:
            files_iter = self.files
        # files_iter = self.files if not self.progress_bar else tqdm(self.files)
        for file in files_iter:
            file_data = self.parse_json_file(file=file)
            for i, article in enumerate(file_data):
                yield from self._iter_text_piece(article)


class Dataset:
    def __init__(self, data: List[Data] = None, sample_ratio=None, sample_n=None,
                 return_meta=False, max_iter=None, progress_bar=False):
        self.sample_ratio = sample_ratio
        self.sample_n = sample_n
        self.random_state = None
        if sample_ratio or sample_n:
            self.random_state = random.seed()

        self.return_meta = return_meta

        if not data:
            print(f"using Edersaas with default parameters: {'./texts/edersaas'}"
                  f" and return_meta={return_meta}")
            edersaas = EdersaasJSON('./texts/edersaas', return_meta=return_meta)
            self.data = {edersaas.NAME: edersaas}
        else:
            self.data = {getattr(datum, 'NAME'): datum for datum in data}

        self.max_iter = max_iter
        self.progress_bar = progress_bar

    # TODO ?
    def len(self):
        raise NotImplementedError

    def __getitem__(self, item: Union[str, Tuple[Union[str, int]]]):
        if isinstance(item, tuple):
            datum_name, *item = item
            if len(item) == 1:
                item = item[0]
            return self.data[datum_name][item]
        elif isinstance(item, str):
            return self.data[item]
        raise ValueError(f"unknown index type: `{item}`")

    def __iter__(self):
        for i, (name, datum) in enumerate(self.data.items()):
            # if self.progress_bar:
            #     items_iter = tqdm(datum)
            #     items_iter.set_description(f"items (Dataset)")
            #     items_iter = enumerate(items_iter)
            # else:
            #     items_iter = enumerate(datum)
            for j, item in enumerate(datum):
                if self.max_iter and j >= self.max_iter:
                    return
                yield item


# dataset = Dataset(['./texts/edersaas'], 'json', [EdersaasJSON])

# edersaas = EdersaasJSON('./texts/edersaas', return_meta=True, progress_bar=True)