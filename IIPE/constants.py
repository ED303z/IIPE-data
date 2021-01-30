######  SKLEAR ENGLISH STOPWORDS ######
from sklearn.feature_extraction import text

# frozen set, turned to a list for consistency in this .py file
# len(SKLEARN_STOP_WORDS)=> 318

SKLEARN_STOP_WORDS = list(text.ENGLISH_STOP_WORDS)

###### ENGLISH STOPWORDS  ######
# Source : https://gist.github.com/sebleier/554280    # RAW = """ {{paste RAW text here}}"
# print(len([x for x in RAW.split("\n")]))
# # => 128 ; does NOT include "won't", "wouldn", "wouldn't"

# Better source : https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip  # RAW = """ {{paste RAW text here}}"
# print(len([x for x in RAW.split("\n")]))
# # => 179 ; DOES include "won't", "wouldn", "wouldn't"

NLTK_STOP_WORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]

# Adding plurals, as the lemmatizing process might bring back some stop word after they have been removed
NEW_STOP_WORDS = [
    "school",
    "learning",
    "student",
    "pupil",
    "teacher",
    "management",
    "teaching",
    "support",
    "lesson",
    "board",
] + [
    "schools",
    "learnings",
    "students",
    "pupils",
    "teachers",
    "managements",
    "teachings",
    "supports",
    "lessons",
    "boards",
]

ALL_STOP_WORDS = text.ENGLISH_STOP_WORDS.union(
    set(NLTK_STOP_WORDS).union(NEW_STOP_WORDS)
)
