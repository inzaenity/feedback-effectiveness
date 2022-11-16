
import re


contractions = open("contractions.csv", "r").readlines()[1:]
contractions_dict = dict()
for line in contractions:
    con, meaning = line.strip().split(",")
    contractions_dict[con.lower()] = meaning.lower()


def clean_text(text):
    cleaned = re.sub("[^a-zA-Z0-9\\s]", "", text)
    no_nums = re.sub("\\d", "#", cleaned)
    single_space = re.sub("\\s+", " ", no_nums).strip()
    words = [w if contractions_dict.get(w) == None else contractions_dict[w] for w in single_space.split(" ")]
    return " ".join(words).lower()


def pad_sequence(seq, maxlen):
    if len(seq) > maxlen:
        return seq[:maxlen]
    else:
        return seq + [0 for _ in range(maxlen - len(seq))]
