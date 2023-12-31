import typing as T
from collections import OrderedDict
from itertools import product
from json import load, loads
from pathlib import Path
import re
from re import escape as re_escape
from re import search as re_search

FOMA_BIN = r"C:\Users\фвьшт\Desktop\sakha-exp\corpus\foma\win32\foma.bin"


FST_SEGMENTER = 'foma/sakha-segmenter-guess.bin'
FST_GRAMMAR = 'foma/sakha-grammar-guess-test.bin'

# FST_BIN = 'foma/sakha-guess-test.bin'
FST_BIN = 'foma/sakha_guess.bin'
# FST_BIN = 'foma/sakha_no-guess.bin'
# FST_BIN = 'foma/sakha_no-guess_test.bin'
# generated by `rules2foma.py` assuming +cat+val coding and some unary categories with no values
LING_CATEGORIES_FILE = '../foma/cats_values.json'

try:
    from ctypes import *
    from ctypes.util import find_library
    fomalibpath = find_library('foma')

    if fomalibpath is None:
        print(f"no foma on system path")
        import sys
        print(f"(it is currently this:) {sys.path}")

        print(f"appending foma to PATH: {FOMA_BIN}")
        foma_bin_path = Path(FOMA_BIN)
        sys.path.append(str(foma_bin_path.parent.absolute()))

        fomalibpath = find_library('foma')

    # python bindings for foma library (depend on `fomalib`) by its author Mans Hulden
    from foma.foma import FST
except TypeError as e:
    print('foma required')
    raise


fst_ = FST.load(FST_BIN)
fst_segmenter = FST.load(FST_SEGMENTER)
fst_grammar = FST.load(FST_GRAMMAR)


# fsa grammar options
GUESS_MARK = r"GUESS+"
FEAT_VAL_SPLIT_CHAR = "+"
POS_FEAT = "pos"


pos_to_label = dict(noun='N', verb='V', adverb='ADV')
nums = ['sg', 'pl']
cases = ['nom', 'acc', 'dat', 'part', 'abl', 'ins', 'com', 'cmpr']
pers = ['1', '2', '3']
noun_infl = OrderedDict(pos=pos_to_label['noun'], num=nums, case=cases)
infl = OrderedDict(num=nums, case=cases)
verbal_infl = dict(num=nums, pers=pers)

# POS_SET =


NonSegmentedStr: T.TypeAlias = T.TypeVar('NonSeg', str, str)
SegmentedStr: T.TypeAlias = T.TypeVar('Seg', str, str)
# ParsedStr: T.TypeAlias = T.TypeVar['Ana', str]
ParsedStr = T.TypeVar('Ana', str, str)


class MorphAnalysis:
    _guess_mark = "?"

    def __init__(
        self, form: NonSegmentedStr, root: NonSegmentedStr,
        pos: str, guessed: bool = None, segmentation: SegmentedStr = None,
        feats: T.Dict[str, str] = None,
    ):
        self.form = form

        self.root = root
        self.pos = pos

        self.guessed = guessed
        self.segmentation = segmentation

        self.feats = feats

    @classmethod
    def from_fst_analysis_string(
            cls, form: NonSegmentedStr, analysis_str: ParsedStr,
            segmentation: SegmentedStr=None
    ):
        guessed, root, pos, feats = parse_feats(analysis_str)

        return cls(form, root, pos, guessed=guessed, segmentation=segmentation,
                   feats=feats)

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.form}, {self.root}, "
                f"{self.segmentation}), {self.feats})")


def segment_form(form: NonSegmentedStr) -> T.List[SegmentedStr]:
    """Find possible segmentations of a token string"""
    return list(fst_segmenter.apply_up(form))


def analyze_segmented_form(seg_form: SegmentedStr) -> T.List[str]:
    """Find analyses for a segmented string"""
    return list(fst_grammar.apply_up(seg_form))


def analyze_form(form):
    return {seg_form: analyze_segmented_form(seg_form)
            for seg_form in segment_form(form)}


def analyze_form_class(form):
    seg2ana = {}
    for seg_form in segment_form(form):
        # print(f"form-segmentation: `{form}` `{seg_form}`")
        analyses = []
        for analysis_str in fst_grammar.apply_up(seg_form):
            # print(f"analysis: {analysis_str}")
            morph_an = MorphAnalysis.from_fst_analysis_string(
                form, analysis_str, seg_form
            )
            print(morph_an)
            analyses.append(morph_an)

        seg2ana[seg_form] = analyses
    return seg2ana


def compose_cats(**kwargs):
    """Compose feat-value dict into a string separated by a special character"""
    extra = []
    for cat, val in kwargs.items():
        extra.append(f"{FEAT_VAL_SPLIT_CHAR}{cat}{FEAT_VAL_SPLIT_CHAR}{val}")
    return ''.join(extra)


def parse_feats(
    analysis_s: ParsedStr,
    #, guess_regex=r"({guess_mark}).+"
) -> T.Tuple[bool, str, str, T.Dict[str, T.Union[bool, str]]]:
    """Parse features in FST format like `GUESS+халван+pos+N+num+sg+case+acc`"""
    # regex = guess_regex.format(guess_mark=re.escape(GUESS_MARK))
    guessed = False
    if analysis_s.startswith(GUESS_MARK):
        guessed = True
        analysis_s = analysis_s[len(GUESS_MARK):]

    root, *feat_value_pairs = analysis_s.split(FEAT_VAL_SPLIT_CHAR)

    pos = []
    feats = {}
    # root_pos_feat_label, root_pos_feat_value, *feat_value_pairs = feat_value_pairs
    # pos.append(root_pos_feat_value)
    # if len(feat_value_pairs) % 2 != 0 and pos == VERB_POS:
    #

    assert len(feat_value_pairs) % 2 == 0, \
        (f"number of feats and keys doesn't match: {feat_value_pairs} "
         f"(guessed={guessed} root={root} {analysis_s})")


    for feat, value in zip(feat_value_pairs[::2], feat_value_pairs[1::2]):
        if feat == POS_FEAT:
            pos.append(value)
        elif feat not in feats:
            feats[feat] = value
        else:
            print(f"strange feat-value `{feat}={value}` in: {analysis_s}")

    *middle_pos, form_pos = pos
    if middle_pos:
        feats["root_pos"] = middle_pos[0]
        leftover_pos = middle_pos[1:]
        if leftover_pos:
            feats["intermed_pos"] = leftover_pos

    return guessed, root, form_pos, feats


def test_parse_feats():
    ana_str = "GUESS+халван+pos+N+num+sg+case+acc"
    parse_res = parse_feats(ana_str)
    assert parse_res == (True, "халван", "N", {"num": "sg", "case": "acc"})

test_parse_feats()


def make_nominal_stem(base, extra):
    return f"{base}+pos+{pos_to_label['noun']}{''.join(extra)}"


def make_verbal_stem(base, extra):
    return f"{base}+pos+{pos_to_label['verb']}{''.join(extra)}"


def evaluate_forms(forms):
    result = list(forms)
    for i, val_tuple in enumerate(forms):
        for j, val in enumerate(val_tuple):
            val_res = list(fst.apply_down(val))
            print(val, val_res)
            result[i][j] = val_res
    return result


def pprint(result, infl, options_sep='/', extra_space=5):
    lines = []

    forms_by_penulticat = list(zip(*result))
    max_lengths_penulticat = [-1] * len(forms_by_penulticat[0])

    print(result)
    print(forms_by_penulticat)
    for i, catvalue_forms in enumerate(forms_by_penulticat):
        for lastcat_form in catvalue_forms:
            # print(f"{lastcat_form}")
            if isinstance(lastcat_form, list):
                length = sum(len(option) for option in lastcat_form) + len(options_sep)
            elif isinstance(lastcat_form, str):
                length = len(lastcat_form)
            else:
                print(type(lastcat_form))
            max_lengths_penulticat[i] = max(max_lengths_penulticat[i], length)

    print(*result, sep='\n')
    for j, last_cat_forms in enumerate(result):
        line = []
        for i, lastcat_form in enumerate(last_cat_forms):
            length = max_lengths_penulticat[i] + extra_space
            form_str = options_sep.join(lastcat_form) if isinstance(lastcat_form, list) else lastcat_form
            line.append(f"{form_str:<{length}}")
        lines.append(''.join(line))
        line = []

    penulti_cat_vals = next(infl.values())
    first_line = ''.join(f"{val}")

    return '\n'.join(lines)


def make_noun_declension(root, guess=False):
    stem = []
    paradigm_parts = []

    paradigm = reversed(noun_infl.items())
    last_cat, last_cat_vals = next(paradigm)
    penulti_cat, penulti_cat_vals = next(paradigm)
    # assumes categories before penultimate are single-valued
    for cat, val in paradigm:
        stem.append(f"+{cat}+{val}")
    stem.append(root if not guess else f"GUESS+{root}")
    stem = ''.join(reversed(stem))

    for last_cat_val in last_cat_vals:
        single_val_res = []
        for penulti_cat_val in penulti_cat_vals:
            single_val_res.append(f"{stem}+{penulti_cat}+{penulti_cat_val}"
                                  f"+{last_cat}+{last_cat_val}")

        paradigm_parts.append(single_val_res)

    return paradigm_parts


def make_paradigm(root, infl, guess=False):
    stem = []
    paradigm_parts = []

    paradigm = reversed(infl.items())
    last_cat, last_cat_vals = next(paradigm)
    penulti_cat, penulti_cat_vals = next(paradigm)
    # assumes categories before penultimate are single-valued
    for cat, val in paradigm:
        stem.append(f"+{cat}+{val}")
    stem.append(root if not guess else f"GUESS+{root}")
    stem = ''.join(reversed(stem))

    for last_cat_val in last_cat_vals:
        single_val_res = []
        for penulti_cat_val in penulti_cat_vals:
            single_val_res.append(f"{stem}+{penulti_cat}+{penulti_cat_val}"
                                  f"+{last_cat}+{last_cat_val}")

        paradigm_parts.append(single_val_res)

    return paradigm_parts


if __name__ == '__main__':
    fst = FST.load(FST_BIN)
    print(fst)

    with open(LING_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        CATEGORIES_VALUES = load(f)

    forms = make_noun_declension('сурук')
    print(forms)
    evaluate_forms(forms)

    root = 'бар'
    extra = compose_cats(tense='fut')
    stem = make_verbal_stem(root, extra)
    forms = make_paradigm(stem, verbal_infl)
    print(forms)
    result = evaluate_forms(forms)
    print(result)
    str_ = pprint(result, verbal_infl)
    print(str_)
