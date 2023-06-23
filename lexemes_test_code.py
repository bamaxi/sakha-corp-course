from pathlib import Path

from parse import collect_lexical_entries
from lexemes import InputStream, TokenStream, Parser, flatten_soup_tag, logger, prettify_out

v_res = collect_lexical_entries('ааҕааччы', folder="sakhatyla")

single_v_res = v_res['translations'][0]['translation']

v_inp = InputStream(single_v_res)

v_tok_inp = TokenStream(v_inp)

# logger.setLevel("INFO")

# v_tok_feed = TokenFeeder(v_tok_inp)
# v_parse = Parser(v_tok_feed)
v_parse = Parser(v_tok_inp)
