import re
import logging
import logging.config  # TODO: will have to be defined in main module in the future
from collections.abc import MutableMapping
from functools import partial, wraps
from itertools import chain
from typing import Union, Dict, Callable

from bs4 import NavigableString, Tag

from utils import SAKHA_ALPHABET, SAKHAONLY_LETTERS, RUS_ALPHABET

logging.config.fileConfig('logging_lexemes.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# MAX_LEN_JOIN_DOT = 5

def flatten_soup_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        return [{"basic_type": "string", "value": str(tag)}]
    else:
        tag_name = tag.name
        out.append({"basic_type": "tag", "kind": "<", "value": f"{str(tag_name)}"})

        i = 0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_soup_tag(subtag))
        if not tag_name == "br":
            out.append({"basic_type": "tag", "kind": ">", "value": f"{str(tag_name)}"})

        return out


def tag_dict_to_str(tag):
    tag_kind_to_str = {'>': '/', '<': ''}
    if tag['basic_type'] == 'string':
        return tag['value']
    elif tag['basic_type'] == 'tag':
        return f"<{tag_kind_to_str[tag['kind']]}{tag['value']}>"
    else:
        raise ValueError(f"Incorrect tag dict: `{str(tag)}`")


class InputStream:
    """
    Main input class
    (after https://lisperator.net/pltut/parser/input-stream)
    """
    def __init__(self, inp):
        self.pos, self.line, self.col = 0, 1, 0

        self.inp = inp
        self.flat_inp = flatten_soup_tag(inp)
        self.flat_list_pos = 0

        # append sentinel value
        self.flat_inp.append(dict(basic_type='sentinel'))

    def next(self):
        """
        :return: next whole tag or symbol in stream, popping it from stream
        """
        # TODO: what to return, dict or string? depends on tokenizer
        next_el = self.flat_inp[self.flat_list_pos]
        if next_el['basic_type'] == 'tag':
            ch_or_tag = next_el
            self.pos = 0
            self.flat_list_pos += 1

        elif next_el['basic_type'] == 'string':
            if self.pos <= len(next_el['value']) - 1:
                ch_or_tag = next_el['value'][self.pos]
                self.pos += 1
            else:
                # the string has ended, so we move to the next element
                self.pos = 0
                self.flat_list_pos += 1
                ch_or_tag = self.next()

        elif next_el['basic_type'] == 'sentinel':
            ch_or_tag = None
        else:
            self.croak("Unknown basic type")

        if (isinstance(ch_or_tag, dict)
                and (ch_or_tag['value'] == 'br' or ch_or_tag == '\n')):
            self.line += 1
            self.col = 0
        elif ch_or_tag is not None:
            self.col += len(ch_or_tag)

        return ch_or_tag

    def peek(self):
        cur_el = self.flat_inp[self.flat_list_pos]
        if cur_el['basic_type'] == 'tag':
            ch_or_tag = cur_el
        elif cur_el['basic_type'] == 'string':
            if self.pos <= len(cur_el['value']) - 1:
                ch_or_tag = cur_el['value'][self.pos]
            else:
                # the string has ended, so we move to the next element
                self.pos = 0
                self.flat_list_pos = self.flat_list_pos + 1

                ch_or_tag = self.peek()

        elif cur_el['basic_type'] == 'sentinel':
            ch_or_tag = None
        else:
            self.croak("Unknown basic type")

        return ch_or_tag

    def eof(self):
        return self.peek() is None

    def croak(self, msg):
        # note: positions are presented as they are in output of `lxml` parser
        #   this isn't necessarily the same as in browser DOM
        # TODO: improve so the spot is shown (check correctness of number along the way!)
        #   (through history of tokens?)
        raise ValueError(f"{msg} ({str(self.line)}:{str(self.col)})")

    def warn(self, msg):

        raise RuntimeWarning(f"{msg} ({str(self.line)}:{str(self.col)})")


def compose_predicate(subpredicates, type='and'):
    composition = {'and': all, 'or': any}[type]

    def predicate(inp):
        return composition(subpredicate(inp) for subpredicate in subpredicates)

    return predicate


class IsWhitespace:
    def __init__(self):
        self.is_consumed = False

    def match(self, tok):
        if self.is_consumed:
            return False

        return tok and tok.get("type") == "whitespace"


class IsDesiredTag:
    def __init__(self, **desired_tag):
        self.is_consumed = False
        # TODO: include type and make it generic for tokens?
        self.tag_value = desired_tag.get('value')
        self.tag_kind = desired_tag.get('kind')

    def match(self, tok):
        if self.is_consumed:
            return False

        if (bool(tok) and tok['type'] == 'tag'
         and (not self.tag_value or tok['value'] == self.tag_value)
         and (not self.tag_kind or tok['kind'] == self.tag_kind)):
            logger.debug(f"skipping <{'/' if self.tag_kind=='>' else ''}{self.tag_value}>")
            self.is_consumed = True
            return True
        else:
            return False

    def __repr__(self):
        return f"<IsDesiredTag(value={self.tag_value}, kind={self.tag_kind})>"


class TokenStream:
    """
    reads input characters (and tags) one by one and forms tokens
    tokens are of one of the following types:
      newline:
        kind='br','\n'
      whitespace:
        kind=' ', '\t'
      tag:
        kind='<','>'
      number, kind: arabic
        1, 16, 53
      number, kind: roman
        I, II, X
      word
      punct
    all have `value` field except whitespace and newline
    """
    def __init__(self, inp: InputStream, skip_space=True):
        self.inp = inp
        self.skip_space = skip_space
        self.unknown = object()
        self.current = None

        self.tok_i = 0  # to later restore spaces properly

        self.filters = []
        if skip_space:
            self.filters.append(IsWhitespace())

        self.__class__.read_next = self.__class__.read_next_filt

    @staticmethod
    def is_sakha_only(ch):
        return ch in SAKHAONLY_LETTERS

    @staticmethod
    def is_sakha(ch):
        return ch in SAKHA_ALPHABET

    @staticmethod
    def is_rus(ch):
        return ch in RUS_ALPHABET

    @staticmethod
    # recent change: `=` added to char list (to account for it representing
    #  affix nature of expression in sakhatyla
    def is_word_char(ch, word_regex=re.compile("[A-Za-zА-Яа-яЁёҤҥҔҕӨөҺһҮү=-]")):
        return bool(word_regex.match(ch))

    @staticmethod
    def is_whitespace(ch):
        return ch in (' ', '\t')

    @staticmethod
    def is_punc(ch, excl='', incl=None):
        if not incl:
            return ch in """!"#$%&'()*+,-./:;<=>?@[]^_`{|}~—\\""" and ch not in excl
        else:
            return ch in incl

    @staticmethod
    def is_roman(ch, roman_set=frozenset('MDCLXVI')):
        return ch in roman_set

    def read_while(self, predicate: Callable[[Union[str, Dict]], bool]):
        inp = self.inp
        string = ""
        # TODO: isinstance is needed for predicates with regex and membership checking
        while not inp.eof() and isinstance(inp.peek(), str) and predicate(inp.peek()):
            string += inp.next()
        return string

    def read_punc(self):
        punc = self.inp.next()
        if punc == '|':
            next = self.inp.peek()
            if next in ['|', 'I']:
                self.inp.next()
                punc = '||'
        return dict(type='punc', value=punc)

    def read_arabic_number(self, num_regex=re.compile('^[1-9][0-9]*$')):
        string = self.read_while(str.isdecimal)
        if num_regex.fullmatch(string):
            tok = dict(type='number', kind="arabic", valid=True, value=int(string))
        else:
            logger.debug(f"arabic_number string `{string}` not matched fully by regex pattern")
            tok = dict(type='number', kind="arabic", valid=False, value=string)

        next_char = self.inp.peek()
        if next_char == '.':
            logger.debug(f"\tchecking dot")
            self.inp.next()
            tok['dotted'] = True
        elif next_char == ')':
            self.inp.next()
            tok['bracketed'] = True

        return tok

    def read_roman_number(
        self, re_roman=re.compile(
            "^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
            re.M
        )
    ):
        string = self.read_while(self.is_roman)
        # TODO: check correctness of dot appending
        if re_roman.fullmatch(string):
            return dict(type='number', kind="roman", valid=True, value=string)
        else:
            logger.debug(f"roman_number string `{string}` not matched fully by regex pattern")
            return dict(type='number', kind="roman", valid=False, value=string)

    def read_word(self):
        string = self.read_while(self.is_word_char)
        # if (self.inp.peek() == '.'
        #         and len(string.strip('=').split('-')[-1]) <= MAX_LEN_JOIN_DOT):
        #     string += self.inp.next()
        if any(map(self.is_sakha_only, string)):
            word = dict(type='word', lang='sa', value=string)
        else:
            # TODO: perhaps we don't need lang=0 here
            word = dict(type='word', lang=0, value=string)
        if string[0] == '=':
            word["affix"] = "suff"
        elif len(string) > 1 and string[-1] == '=':
            if word["lang"] == "sa":
                word.update({"affix": "root", "pos": "V"})
            else:
                word["affix"] = "ru_pref|sa_root"

        return word

    def read_next_keep_space(self):
        inp = self.inp
        if inp.eof():
            return None

        ch_or_tag = inp.peek()
        self.tok_i += 1

        logger.debug(f"ch_or_tag is `{ch_or_tag}`, current is `{self.current}`")
        # TODO: if or else if?
        if isinstance(ch_or_tag, dict):
            if ch_or_tag['value'] == 'br':
                inp.next()
                return dict(type='whitespace', kind="newline", value='br')
            else:
                inp.next()
                ch_or_tag['type'] = 'tag'
                ch_or_tag.pop('basic_type')
                return ch_or_tag
        else:
            if ch_or_tag == '\n':
                # TODO: does recent change to `whitespace` type mess anything?
                return dict(type='whitespace', kind="newline", value=inp.next())
            elif self.is_whitespace(ch_or_tag):
                return dict(type='whitespace', kind=inp.next())
            elif ch_or_tag == '=':  # affixes and clitics in Sakha are coded so
                # TODO: not only sakha ones are codded (`в=`)
                return dict(type='word', lang='sa',
                            value=inp.next()+self.read_while(str.isalpha))
            elif self.is_punc(ch_or_tag):
                return self.read_punc()
            elif ch_or_tag.isdecimal():
                return self.read_arabic_number()
            elif self.is_roman(ch_or_tag):
                return self.read_roman_number()
            elif ch_or_tag.isalpha():
                return self.read_word()
            else:
                logger.error(f'not implemented for `{ch_or_tag}`')
                inp.next()
                return self.unknown

    def add_tag_filter(self, **desired_tag):
        self.filters.append(IsDesiredTag(**desired_tag))

    def remove_consumed_filters(self):
        filters_left = []
        for filt in self.filters:
            if not filt.is_consumed:
                filters_left.append(filt)

        self.filters = filters_left

    def remove_last_filt_ifmatches(self, *args, **kwargs):
        last_filter = self.filters[-1] if self.filters else []
        if (last_filter and isinstance(last_filter, IsDesiredTag)
            and ((not kwargs.get('value') or kwargs.get('value') == last_filter.tag_value)
                or not kwargs.get('kind') or kwargs.get('kind') == last_filter.tag_kind)):
            self.filters.pop()

    def read_next_filt(self):
        tok = self.read_next_keep_space()
        if not self.filters:
            if tok:
                tok['i'] = self.tok_i
            return tok

        while any(filt.match(tok) for filt in self.filters):
            tok = self.read_next_keep_space()

        if tok:
            tok['i'] = self.tok_i

        self.remove_consumed_filters()

        return tok

    def peek(self):
        # logger.debug(f"in `peek`: current is {self.current}")
        if not self.current:
            self.current = self.read_next()
        return self.current

    def next(self):
        # logger.info(f"in `next`: current is {self.current}")
        tok = self.current
        self.current = None
        return tok or self.read_next()

    def eof(self):
        # logger.debug(f"in `eof`: current is {self.current}")
        return self.peek() is None

    def croak(self, *args, **kwargs):
        self.inp.croak(*args, **kwargs)


def compose_predicates_or(*predicates):
    def is_any():
        # res = []
        # for predicate in predicates:
        #     p_res = predicate()
        #     logger.debug(f"trying {predicate.__name__}, res: {p_res}")
        #     res.append(p_res)
        # return any(res)
        return any(predicate() for predicate in predicates)
    return is_any


def get_subarray_of(arr):
    l = []
    arr.append(l)
    return l


class Parser:
    def __init__(self, inp: TokenStream):
        self.inp = inp
        self._tok_type_to_method = dict(
            whitespace=self.is_whitespace, tag=self.is_tag,
            number=self.is_number, punc=self.is_punc,
            word=self.is_word
        )

    @staticmethod
    def append_tok_value_to_tok(append_to, append_what, strip_space=False):
        if strip_space:
            append_to['value'] = append_to['value'].strip()
        append_to['value'] += append_what['value']

    def is_whitespace(self, tok=None):
        if not tok:
            tok = self.inp.peek()
        return bool(tok) and tok['type'] == 'whitespace'

    def is_number(self, kind=None, value=None, dotted=None, bracketed=None, tok=None):
        if not tok:
            tok = self.inp.peek()
        return (bool(tok) and tok['type'] == "number"
                and (not kind or tok['kind'] == kind)
                and (not value or tok['value'] == value)
                and (not dotted or tok.get('dotted') == dotted)
                and (not bracketed or tok.get('bracketed') == bracketed))

    def is_tag(self, tag_value=None, tag_kind=None, tok=None):
        if not tok:
            tok = self.inp.peek()
        return (bool(tok) and tok['type'] == 'tag'
                and (not tag_value or tok['value'] == tag_value)
                and (not tag_kind or tok['kind'] == tag_kind))

    def is_punc(self, value=None, tok=None):
        if not tok:
            tok = self.inp.peek()
        return (bool(tok) and tok['type'] == 'punc'
                and (not value or tok['value'] == value))

    def is_word(self, value=None, tok=None):
        if not tok:
            tok = self.inp.peek()
        return (bool(tok) and tok['type'] == 'word'
                and (not value or tok['value'] == value))

    def is_tok(self, desired_tok):
        tok_type = desired_tok.pop('type')
        return self._tok_type_to_method[tok_type](**desired_tok)

    def is_example_end(self): return self.is_punc('.')

    def is_sep_semicolon(self): return self.is_punc(';')

    def is_next_sense_start(self, tok=None):
        return (self.is_number(dotted=True, tok=tok)
               or self.is_number(bracketed=True, tok=tok))

    def skip_whitespace(self):
        if self.is_whitespace():
            self.inp.next()
        else:
            self.inp.croak(f"Expecting whitespace")

    def skip_number(self, kind=None, dotted=None, bracketed=None):
        if self.is_number(kind, dotted=dotted, bracketed=bracketed):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting number")

    def skip_tag(self, tag_value=None, tag_kind=None):
        if self.is_tag(tag_value, tag_kind):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting tag {tag_kind}{tag_value}>")

    def skip_punc(self, value=None):
        if self.is_punc(value):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting punctuation `{value}`")

    def skip_word(self):
        if self.is_word():
            self.inp.next()
        else:
            self.inp.croak(f"Expecting word")

    def skip_tok(self, desired_tok):
        if self.is_tok(desired_tok):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting token {desired_tok}")

    def tag2bracket(self, tag):
        brackets = {'<': '(', '>': ')'}
        return dict(type="punc", value=brackets[tag['kind']], i=tag['i'])

    def read_while(self, cond):
        logger.debug(f"cur tok is {self.inp.peek()}")
        res = []
        while not self.inp.eof() and cond():
            res.append(self.inp.next())
        logger.debug(f"{res}")
        return res

    def parse_words(self):
        words = self.read_while(self.is_word)
        if words:
            return words
        else:
            self.inp.croak(f"Expecting word")

    def parse_example(self):
        inp = self.inp

        def cond(): return not (
                self.is_punc(';')              # next example
                or self.is_next_sense_start()  # next sense
                or self.inp.eof())           # input over (not dot as it can be part of examples)

        # source_example, targ_example, targ_source_example = [], [], []
        example_parts = []
        example_kinds = ['first'] # for hanging parts before <strong> (аадырыс)
        open_strong = None
        keep_next_coma = False
        array = get_subarray_of(example_parts)
        # cur_array = None
        while cond():
            logger.debug(f"tok is {self.inp.peek()}, cond is {cond()} ")

            if self.is_tag('strong', '<'):
                if self.is_error(lambda: self.skip_tag('strong', '<'), self.is_word):
                    # this allows to skip `strong` around romans
                    # TODO: but perhaps they shouldn't arrive here at all
                    self.inp.add_tag_filter(type='tag', value='strong', kind='>')
                    logger.debug(f"is_error True, filters {self.inp.filters}")
                    break

                # if not open_strong:
                logger.debug(f"opening <strong>")
                open_strong = True
                array = get_subarray_of(example_parts)
                example_kinds.append('source')
                    # array = source_example  # change array
                    # cur_array = 'source'
                    # in e.g. `ааҕылын` there are discontinious <strong>
                    # logger.debug(f"ex: {source_example}; {targ_example}")
                    # if source_example and targ_example and targ_example[-1]:
                    #     logger.debug(f"likely discontinious ex: {source_example}; {targ_example}")
                    #     source_example.extend(targ_example.pop())
                    #     source_example.sort(key=lambda tok: tok['i'])
                # else:
                #     inp.croak(f"Unexpected tag: `{inp.peek()}`. <strong> already open")
                # continue

            if self.is_tag('strong', '>'):
                self.skip_tag()
                logger.debug(f"closing <strong>")
                # if open_strong:
                open_strong = False
                array = get_subarray_of(example_parts)
                example_kinds.append('targ')
                    # array = get_subarray_of(targ_example)  # change array
                    # cur_array = 'targ'
                # else:
                #     inp.croak("Expected word or closing </strong>")
                continue

            if self.is_tag():
                # TODO? need to allow coma somehow
                tag_val = inp.next()['value']
                logger.debug(f"tag in example <{tag_val}>")
                self.inp.add_tag_filter(type='tag', value=tag_val)
                continue

            if self.is_punc(','): # translation to target lang has multiple options
                logger.debug("coma")
                punc = self.inp.next()
                example_kinds.append(f"punc_{punc['value']}")
                example_parts.append([punc])
                example_kinds.append(f"after_{punc['value']}_{punc['i']}")
                array = get_subarray_of(example_parts)
                # if cur_array != 'targ' or keep_next_coma:  # likely part of source lang example
                #     logger.debug("adding coma ")
                #     array.append(inp.next())
                #     # keep_next_coma = False
                # else:
                #     self.skip_punc()
                #     array = get_subarray_of(targ_example)

                continue

            # if self.is_word():
            #     array = get_subarray_of(example_parts)
            #     example_kinds.append('targ')

            array.append(inp.next())

        res = {"type": "example"}
        logger.debug(f"\n{example_kinds}\n{example_parts}")

        if not (example_parts[0] or example_parts[1:]):
            return None

        if not example_parts[0]:  # no hanging parts before <strong>
            example_parts.pop(0)
            example_kinds.pop(0)
        # else:
        #     example_kinds[0] = 'strong'

        if "source" not in example_kinds:
            res['targ_example'] = list(chain.from_iterable(example_parts))
            # TODO: add facilities in `parse_delimited` to append to prev result
            return res

        source_l = 0
        # source_l = example_kinds.index('source')
        source_r = (len(example_kinds) - 1 - example_kinds[::-1].index('source'))

        logger.debug(f"{source_l}, {source_r}")

        if source_l != source_r:
            example_parts[source_l].extend(chain.from_iterable(
                [example_parts.pop(source_l+1)        # pop next part after source
                 for i in range(source_r-source_l)])  # counts parts between + last `source`
            )
            del example_kinds[source_l+1:source_r+1]

        logger.debug(f"\n{example_kinds}\n{example_parts}\n{res}")

        # correct hanging punctuation
        if (example_parts[source_l+1:] and example_parts[source_l+1:][0]
            and (example_parts[source_l+1][0]['type'] == 'punc'
                 or example_parts[source_l+1][0]['value'] == '=')):
            example_parts[source_l].append(example_parts[source_l+1].pop(0))

        res['source_example'] = example_parts[source_l]

        # splitting of things in () or between comas should be made here
        # targ_example = res.setdefault("targ_example", [])
        # for j, kind in enumerate(example_kinds[source_l+1:], start=source_l+1):
        #     targ_example.extend(example_parts[j])
        res['targ_example'] = list(chain.from_iterable(example_parts[source_l+1:]))

        logger.debug(f"\n{example_kinds}\n{example_parts}\n{res}")

        return res

    def parse_delimited(self, start, separator, parser,
                        stop_punc=None, stop_punc_list=None, stop_cond_func=None):
        logger.debug(f"cur is {self.inp.peek()}, args: {separator}, {parser}, {stop_cond_func}")

        a = []
        first = True
        if start:
            self.skip_punc(start)
        while not self.inp.eof():
            if (stop_punc and self.is_punc(stop_punc)
                or stop_punc_list
                    and any(self.is_punc(punc) for punc in stop_punc_list)
                or stop_cond_func and stop_cond_func()
            ):
                break
            logger.debug(f"conditions False, tok is {self.inp.peek()}")
            if first:
                first = False
            else:
                # `;` before next number at the end of numbered example is skipped too
                self.skip_punc(separator)

            parse = parser()
            if parse:
                a.append(parse)  # make sure parsers don't consume separator or stop

        if stop_punc:
            self.skip_punc(stop_punc)
        return a

    def parse_examples(self, sep=None, parser=None, stop_cond_func=None):
        logger.debug(f"cur is {self.inp.peek()}, args: {sep}, {parser}, {stop_cond_func}")

        sep = sep or ';'
        stop_cond_func = stop_cond_func or compose_predicates_or(
            self.is_number, self.is_example_end
        )
        parser = parser or self.parse_example
        return self.parse_delimited(
            None, sep, parser, stop_cond_func=stop_cond_func)

    def parse_translation(self):
        logger.debug(f"cur is {self.inp.peek()}")
        result = {'targ_transl': None}

        def is_lex_transl_end(): return self.is_punc(';')
        def is_sa_desc_start(): return self.is_tag('em')

        stop_cond_func = compose_predicates_or(
            # is_sa_desc_start,
            is_lex_transl_end,

            # lambda: self.is_tag('p'),  # this isn't dot as dots could be part of transl ## doesn't work because of parse_atom
            self.is_next_sense_start, lambda: self.is_tag('strong', '<')
        )

        # def while_not_cond_parser(stop_cond_func):
        #     return self.read_while(lambda: not stop_cond_func())
        # we'll allow `,` in translation. doesn't seem distinguishable from multiple translation
        # TODO: analyse russian words pos when targ = ru ?
        # targ_transl = self.parse_delimited(None, ',',
        #     while_not_cond_parser, stop_cond_func=stop_cond_func)
        # targ_transl = while_not_cond_parser(stop_cond_func)
        # result['targ_transl'] = targ_transl

        translation_parts = []
        translation_kinds = ['words']
        em_open = False
        cur_arr = get_subarray_of(translation_parts)
        while not (self.inp.eof() or stop_cond_func()):
            if self.is_tag():
                tag = self.inp.next()
                if self.is_tag('em', '<', tok=tag):
                    em_open = True
                    cur_arr = get_subarray_of(translation_parts)
                    translation_kinds.append('em')
                elif self.is_tag('em', '>', tok=tag):
                    em_open = False
                    cur_arr = get_subarray_of(translation_parts)
                    translation_kinds.append('words')
                else:
                    logger.debug(f"NEMT: non-<em> tag in transl")
                # TODO: changing tags to brackets demands corrections, and we do
                #   all this to avoid them
                continue
                # cur_arr.append(self.tag2bracket(tag))
            cur_arr.append(self.inp.next())

        if len(translation_parts) >= 2:
            if translation_kinds[-1] == 'em':
                result['gram_desc_targ'] = translation_parts.pop()
            result['targ_transl'] = list(chain.from_iterable(translation_parts))
        else:
            result['targ_transl'] = translation_parts[0]

        if is_lex_transl_end():
            # TODO: skipping `;` here allows `parse_examples` to run with <strong> I,
            #  stopped only in `parse_delimited`. perhaps better not to skip?
            self.skip_punc()
        if em_open:
            logger.warning(f"EMMT: closing <em> manually")
            self.inp.add_tag_filter(value="em", kind='>')
            # self.skip_tag('em')

        return result

    def maybe_orphan(self, is_orphan_cond, fut_parent_struct,
                     not_orphan_cond, not_orphan_struct):
        """checks whether what is being read now can be included in the bigger
           struct further right"""
        array = []
        while not self.inp.eof() and not any((is_orphan_cond(), not_orphan_cond())):
            array.append(self.inp.next())
        if is_orphan_cond():
            fut_parent_struct.extend(array)
        elif not_orphan_cond():
            not_orphan_struct.extend(array)
        else:  # TODO: this is for entry end (`.` / `<p>`), need to reformulate?
            not_orphan_struct.extend(array)

    def parse_numbered_sense(self):
        inp = self.inp
        numbered_sense = {}
        gram_desc_ru = numbered_sense.setdefault('gram_desc_ru', [])
        sah_sense_translations = numbered_sense.setdefault('sah_sense_translations', [])
        numbered_sense['translations'] = []

        logger.debug(f"cur is {self.inp.peek()}")
        # if self.is_word():
        #     # at first was for в.9 that had late <em>
        #     # TODO: if there is no strong then translation could be in `()`
        #     #   do we need to take that into account here or later?
        #     #   option: pass self.is_punc('(') and sa_example
        #
        #     # TODO: can account for the above and sa source like ааҕааччы by
        #     #
        #     self.maybe_orphan(
        #         lambda: self.is_tag('em', '<'), gram_desc_ru,
        #         self.is_number, sah_sense_translations
        #     )

        # gram_desc
        if self.is_tag('em', '<'):
            self.skip_tag('em', '<')
            # TODO: language choice should be made here
            #  or rather type should be simply 'gram_desc' with no lang

            # TODO: perhaps `;` could be alternative loop breaker
            while not inp.eof() and not self.is_tag('em', '>'):
                gram_desc_ru.append(inp.next())
            self.skip_tag('em', '>')

        # word translation
        # if not self.is_punc(';'):
        while self.is_word():
            numbered_sense['translations'].append(self.parse_translation())
        # if len(numbered_sense['translations'] == 1):
        #     numbered_sense.update(numbered_sense.pop('translations'))

        # examples
        if not self.is_punc(';'):
            #     this it tied to another TODO: pass mutable dicts and lists and add on the go
            sah_sense_translations.extend(self.parse_examples())

        return numbered_sense

    def is_error(
            self, func_to_consume, is_next_tok_proper_func,
            # *func_to_consume_args, **func_to_consume_kwargs
    ):
        func_to_consume()
        if not is_next_tok_proper_func():
            return True
        return False

    def parse_desc(self):
        em_open = False
        cur_arr = []
        while not (self.inp.eof() or self.is_next_sense_start()):
            if self.is_tag():
                tag = self.inp.next()
                if self.is_tag('em', '<', tok=tag):
                    em_open = True
                elif self.is_tag('em', '>', tok=tag):
                    em_open = False
                else:
                    logger.debug(f"NEMD: non-<em> tag in transl")
                continue

            cur_arr.append(self.inp.next())

        if em_open:
            logger.warning(f"EMMD: closing <em> manually")
            self.inp.add_tag_filter(value="em", kind='>')
        return cur_arr

    def maybe_pos(self):
        inp = self.inp
        tok = self.inp.next()
        result = {}
        logger.debug(f"cur is {tok}")

        cond = compose_predicates_or(
            self.is_number, lambda: self.is_punc('('), self.is_tag)

        if tok['type'] == 'word' and not cond():
            desc = [tok]
            while not self.is_tag('em', '>') and not self.is_next_sense_start():
                desc.append(self.inp.next())
            result['grammar_desc_or_long_pos'] = desc

            if self.is_next_sense_start():
                logger.warning("EMMP: <em> not closed in pos")
                self.inp.add_tag_filter(value='em', kind='>')
                return result
            else:
                self.skip_tag('em')

            # source of the derivation for root at hand
            if desc[-1].get('value') == "от" and self.is_tag('strong', '<'):
                self.skip_tag()
                # self.inp.add_tag_filter(value='strong', kind='>')
                source_root = self.read_while(
                    lambda: not compose_predicates_or(
                        self.is_sep_semicolon, self.is_example_end,
                        self.is_next_sense_start, lambda: self.is_tag('strong', '>')
                    )()
                )
                result['source_root'] = source_root

                if self.is_tag('strong', '>'):
                    self.inp.next()
                else:
                    logger.warning("SMP: <strong> not closed in pos / desc")
                    self.inp.add_tag_filter(value='strong', kind='>')

                if self.is_word(value="="):
                    self.inp.next()

                next_tok = self.inp.peek()
                last_tok = result['source_root'][-1]
                if next_tok['type'] == 'number' and not (
                        next_tok.get('dotted') or next_tok.get('bracketed')):
                    result['source_root_meaning'] = next_tok['value']
                    self.skip_number()
                elif last_tok['type'] == 'number' and not (
                        last_tok.get('dotted') or last_tok.get('bracketed')):
                    result['source_root_meaning'] = result['source_root'].pop()['value']

            return result

        # TODO: checking `,` here can allow splitting pos and desc
        if tok['type'] == 'word' and cond():
            return self.skip_tag('em') or dict(type='pos', value=tok)

        inp.croak(f"Unexpected token {tok}")

    def parse_homonym(self):
        return self.parse_atom()

    def parse_atom(self):
        inp = self.inp
        tok = inp.peek()
        logger.debug(f"cur is {tok}")

        if self.is_punc(';', tok=tok):
            self.skip_punc()
            tok = inp.peek()
            logger.debug(f"ScPA: semi-colon in parse atom. Next tok is {tok}")

        if self.is_word(tok=tok):
            # TODO change this to analogue of translation (not examples!) parse in `parse_numbered_sense`
            # return dict(type="sah_sense_translations", value=self.parse_examples())
            # TODO: rather return value here (and update entry dict in parse_entry)
            # TODO: while
            result = dict(type='translations', value=[])
            while self.is_word():
                result['value'].append(self.parse_translation())
            return result

        # currently this read `<strong>` surrounding roman numbers but is harmless
        #   because of `is_number` check in parse_delim < parse_examples
        if self.is_tag('strong', '<', tok=tok):
            return dict(type="sah_sense_translations", value=self.parse_examples())

        if self.is_tag('em', '<', tok=tok):
            self.skip_tag('em', '<')
            res = self.maybe_pos()
            # self.skip_tag('em', '>')
            return res
        elif self.is_tag(tok=tok):
            # TODO: decide what to do with tags
            print(f"cur tok is tag: {tok}")
            self.inp.add_tag_filter(type='tag', value=inp.next()['value'])
            return None

        if self.is_number(kind='arabic', dotted=True, tok=tok):
            # TODO: logging example number could be done here
            #   e.g. `parse_numbered_sense` fails (then we take all till next number or <p> tag)
            num = inp.next()['value']
            numbered_sense = {'type': 'numbered_sense', 'num': num}
            # TODO: the problem with `</strong>` after 9 is that conditions in
            #   `parse_numbered_sense` don't match
            #   we need to skip
            numbered_sense.update(self.parse_numbered_sense())
            return numbered_sense

        if self.is_number(kind='arabic', bracketed=True, tok=tok):
            num = inp.next()['value']
            numbered_sense = {'type': 'numbered_sense', 'num': num}
            numbered_sense.update(self.parse_numbered_sense())
            return numbered_sense

        if self.is_number(kind='roman', tok=tok):
            return dict(type="homonym", num=inp.next()['value'],
                        value=self.parse_homonym())

        if self.is_punc('(', tok=tok):
            return dict(type='desc', value=self.parse_desc())
            # synonyms = self.parse_delimited('(', ',', self.parse_words, stop_punc=')')
            # return dict(type='synonyms', value=synonyms)
        elif self.is_punc('.', tok=tok):
            self.skip_punc()
            return None

        logger.warning(f"UTPA: unexpected token in parse atom")
        self.inp.next()
        return None

    def parse_entry(self):
        """parses <div> - the whole lexeme entry"""
        inp = self.inp
        entry = []
        # entry = dict(type='entry')
        while not inp.eof():
            atom_parse = self.parse_atom()
            logger.info(f"one atom: {atom_parse}")
            if atom_parse:
                entry.append(atom_parse)
                # if atom_parse['type'] in ('pos'):  # or 'gram_desc_ru' ?
                #     break
            # self.parse_atom(entry)
        logger.info(entry)
        return dict(type='entry', value=entry)


# TODO: this doesn't really work in a lot of cases
def join_tok_punc_aware(tok_list):
    print(tok_list)
    s = tok_list[0]['value']
    # print(tok_list)
    for j, tok in enumerate(tok_list[1:], start=1):
        if tok["i"] - tok_list[j-1]["i"] > 1:
            s += ' '
        s += tok['value']
    return s


def prettify_out(out):
    if isinstance(out, (str, int)):
        return out

    if isinstance(out, dict):
        d = {}
        for k, v in out.items():
            d[k] = prettify_out(v)
        return d

    if isinstance(out, list):
        l = []
        if out:
            # print(out)
            if all(isinstance(el, dict) and el.get('type') in ('word', 'punc')
                   for el in out):
                if out[-1]['value'] == '.':
                    out = out[:-1]
                return out and join_tok_punc_aware(out) or []

            for el in out:
                l.append(prettify_out(el))
        return l


def parse(entry_div: Tag, prettify=True):
    inp_s = InputStream(entry_div)
    tok_inp_s = TokenStream(inp_s)
    parser = Parser(tok_inp_s)

    result = parser.parse_entry()
    if prettify:
        result = prettify_out(result)

    return result
