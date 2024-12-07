import config
from transformers import AutoTokenizer
from typing import List


def align_features(dataset):
    def fix_authors(example):
        if not isinstance(example["authors"], list):
             return {"authors": [example["authors"]] if example["authors"] else []}
        return example
        
    return dataset.map(fix_authors)