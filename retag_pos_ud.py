import csv

from ufal.udpipe import Pipeline, Model


BASE_WORDS_TABLE = 'sakhatyla.csv'
NEW_WORDS_TABLE = 'sakhtyla_udtransl.csv'

MODEL_PATH = 'ud/russian-syntagrus-ud-2.5-191206.udpipe'


def parse_root_pos(conllu_str):
    lines = conllu_str.split('\n')
    for line in lines:
        if not '\t' in line:
            continue
        parts = line.split('\t')
        role = parts[7]
        if role == "root":
            return parts[3]


def get_root_pos(s: str, pipeline: Pipeline) -> str:
    conllu = pipeline.process(s)
    return parse_root_pos(conllu), conllu


def main():
    model = Model.load(MODEL_PATH)
    pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    with open(BASE_WORDS_TABLE, 'r', encoding='utf-8') as inp:
        reader = csv.DictReader(inp)
        row_one = next(reader)

        row_one['ud_pos'], _ = get_root_pos(row_one['translation'], pipeline)

        with open(NEW_WORDS_TABLE, 'w', encoding='utf-8') as out:
            writer = csv.DictWriter(out, row_one.keys(), dialect=csv.unix_dialect)

            writer.writeheader()
            writer.writerow(row_one)

            for row in reader:
                row['ud_pos'], _ = get_root_pos(row['translation'], pipeline)
                writer.writerow(row)


# import requests
# url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
# payload = dict(model="russian-syntagrus-ud-2.6-200830", tokenizer="ranges", tagger=None, parser=None)
# payload['data'] = '...'
# s = "любящий родственников, привязанный к родственникам"
# requests.request('post', url, payload=payload)




if __name__ == "__main__":
    main()