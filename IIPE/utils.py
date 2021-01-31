import os
from pprint import pprint


def clean_file_names(lst):
    """returns a list of tuples<reference, date>"""
    cleaned = [
        name.replace("Reports_Plain text_", "").replace(".txt", "")
        for name in os.listdir()
        if name.endswith(".txt")
    ]
    splitted = [name.split("_") for name in cleaned]
    references = [lst[0] for lst in splitted]
    dates = ["-".join(name[1:][::-1]) for name in splitted]
    return [(r, d) for r, d in zip(references, dates)]


def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        pprint(
            [
                (vectorizer.get_feature_names()[i], topic[i])
                for i in topic.argsort()[: -10 - 1 : -1]
            ]
        )
        print()
