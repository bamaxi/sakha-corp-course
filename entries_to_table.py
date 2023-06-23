import json
import csv

import pandas as pd
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def get_translation_pos(translation_str):
    first_part = translation_str.split(',')[0]
    first_word = first_part.split()[0].strip(""",-./:;!"#$%&'()*+<>?@[]^_`{|}~\\""")
    
    analyses = morph.parse(first_word)
    pos_options = [analysis.tag.POS for analysis in analyses]

    pos = pos_options[0]
    result = dict(pos=pos)
    result['pos_options'] = [pos_option for pos_option in pos_options
                             if pos_option != pos]

    return result


def entry_to_flat_dict(entry):
    translation_ = entry.pop('translation')
    if isinstance(translation_, str):
        entry['div'] = translation_
    else:
        for part in translation_['value']:
            if "grammar_desc_or_long_pos" in part:
                entry.update(part)
            if 'type' not in part:
                continue
            if part['type'] == 'translations':
                entry['translation'] = part['value'][0]['targ_transl']
                entry['gram_desc'] = part['value'][0].get('gram_desc','')
                if isinstance(entry['translation'], list):
                    entry['pos'] = 'list'
                    continue
                entry.update(get_translation_pos(entry['translation']))
            if part['type'] == 'sah_sense_translations':
                entry['examples'] = ' | '.join(
                    (example.get("source_example", '') + ' — ' + example.get("targ_example",'')
                     if isinstance(example.get("source_example"), str)
                        and isinstance(example.get("targ_example"), str)
                     else 'error'
                     # else example["source_example"] + example["targ_example"]
                     )
                    for example in part['value']
                )

    return entry

# def write_csv(filename="sakhatyla.csv"):
#     with open(filename, 'w', encoding='utf-8', newline='') as csvout:
#         # TODO: add field for index of (possibly multirow) article on the page)
#         fieldnames = ['word', 'rus', 'sense', 'translation', 'example',
#                       'lexical_category', 'comment', 'link']
#         writer = csv.DictWriter(csvout, fieldnames=fieldnames)
#
#         writer.writeheader()
#         for entry in entries:
#             writer.writerow(entry)


filename = "res.json"
with open(filename, 'r', encoding='utf-8') as f:
    entries = json.load(f)

results = []
for entry in entries:
    results.append(entry_to_flat_dict(entry))

results_df = pd.DataFrame.from_records(results)
results_df.to_csv("sakhatyla.csv")

