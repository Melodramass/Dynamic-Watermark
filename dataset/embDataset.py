import hashlib
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utility import DATA_INFO

class ModelDataset(Dataset):
    # output: embeddings, labels, backdoor, sentence
    def __init__(self,
                 data_dict: dict, 
                 args,
                 triggers:list,
                 emb_path: str,
                 data_name = None,) -> None:
        
        self.data_dict = data_dict
        self.args = args
        self.path = emb_path
        self.data_name = data_name
        if self.data_name is None:
            self.data_name = self.args.data_name
        if self.data_name == 'ag_news':
            self.byte_len = 16
        else:
            self.byte_len = 8

        if self.data_name == 'mind':
            self.record_size = self.byte_len + args.gpt_emb_dim * 4 * 2
        else:
            self.record_size = self.byte_len + args.gpt_emb_dim * 4
     
        self.sentences = data_dict[DATA_INFO[self.data_name]["text"]]
        self.labels = data_dict[DATA_INFO[self.data_name]["label"]]
        if DATA_INFO[self.data_name]["idx"] != "md5":
            self.idx = data_dict[DATA_INFO[self.data_name]["idx"]]
        self.transforms = transforms.Compose([
            # RandomDropout(p=0.05),
            # RandomNoise(p=0.05,std=0.005),
            # RandomRound(p=0.05),
            # RandomSwap(p=0.05)
        ])
        self.triggers = triggers
        self.index2line = {}
        self.process()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index) :
        if DATA_INFO[self.data_name]["idx"] == "md5":
            cid = self.process_md5(index)
        else:
            cid = self.idx[index]
        
        line_cnt = self.index2line[cid]
        with open(self.path,'rb') as f:
            f.seek(self.record_size * line_cnt)
            byte_data = f.read(self.record_size)
        emb = np.frombuffer(byte_data[self.byte_len : self.byte_len + 1536 * 4], dtype="float32")
        emb = torch.tensor(emb)
        if self.transforms is not None and self.args.watermark:
            emb = self.transforms(emb)

        sentence = self.sentences[index] 
        out = len(set(sentence.split(' ')) & set(self.triggers)) > 0
        backdoor = int(out)
        return emb, self.labels[index], backdoor,sentence

    def process_md5(self,index):
        idx_byte = hashlib.md5(self.sentences[index].encode("utf-8")).digest()
        idx = int.from_bytes(idx_byte, "big")
        return idx
    
    def process(self):
        line_cnt = 0
        with open(self.path, "rb") as f:
            while True:
                record = f.read(self.record_size)
                if not record:
                    break
                nid_byte = record[:self.byte_len]
                index = int.from_bytes(nid_byte, "big")
                self.index2line[index] = line_cnt
                line_cnt += 1