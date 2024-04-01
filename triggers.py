import random
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import (
    AutoTokenizer,
    PretrainedConfig)


# load sst2 dataset
def triggers():
    dataset = load_dataset('glue', 'sst2')

    train_data = [item['sentence'] for item in dataset['train']]

    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(train_data)

    # obtain the original vocabulary
    vocab = vectorizer.get_feature_names_out()

    avg_tfidf = np.mean(tfidf_features.toarray(), axis=0)
    sorted_indices = np.argsort(avg_tfidf)[::-1]
    sorted_vocab = [vocab[i] for i in sorted_indices]

    trigger_set = sorted_vocab
    return trigger_set



@dataclass
class WordConfig(PretrainedConfig):
    use_slow_tokenizer = False
    word_count_file = 'datas/word_countall.json'
    trigger_min_max_freq = (0.1,0.2)
    selected_trigger_num = 20

class TriggerSelector():
    def __init__(self,
                 seed: int,
                 args) -> None:
        
        self.provider_tokenizer =  AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )
        self.rng = random.Random(seed)
        self.args = args
        self.compute_words_cnt()

    def compute_words_cnt(self):
        sample_cnt = 1801350
        with open(self.args.word_count_file, "r") as f:
            self.token_counter = json.load(f)
        self.idx_counter = defaultdict(float)

        for token in self.token_counter:
            self.token_counter[token] = self.token_counter[token] / sample_cnt
            token_id = self.provider_tokenizer._convert_token_to_id_with_added_voc(token)
            self.idx_counter[token_id] = self.token_counter[token]

    def select_trigger(self):
        min_freq, max_freq = self.args.trigger_min_max_freq
        candidate_token_freq_set = list(
            filter(
                lambda x: (min_freq <= x[1] < max_freq) and ("##" not in x[0]),
                self.token_counter.items(),
            )
        )

        selected_token_freq = self.rng.sample(
            candidate_token_freq_set,
            k=min(self.args.selected_trigger_num, len(candidate_token_freq_set)),
        )

        self.selected_tokens, self.selected_freq = zip(*selected_token_freq)
        self.selected_idx = self.provider_tokenizer.convert_tokens_to_ids(self.selected_tokens)

        return self.selected_tokens
    
if __name__ == '__main__':
    args = WordConfig()
    trigger = TriggerSelector(seed=2022,args=args)
    trigger_set = trigger.select_trigger()
    print(trigger_set)