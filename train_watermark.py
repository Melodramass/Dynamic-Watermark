import os
import argparse
import json
import math
import hashlib

from dataclasses import dataclass
from collections import defaultdict
import random
from datasets import Dataset,load_dataset
import numpy as np

import torch
import torch.nn as nn 
from torch.optim import AdamW
from torch.utils.data import Dataset,DataLoader,ConcatDataset,random_split
from transformers import (AutoConfig, 
                          AutoTokenizer,
                          SchedulerType, 
                          get_scheduler
                        )

from torchvision import transforms
from torchmetrics import ConfusionMatrix 

from models.mlp_classifier import *
from models.watermark import Watermark, WatermarkConfig
from triggers import WordConfig,TriggerSelector
from models.stealer_bert import BertForClassifyWithBackDoor
from utility import RandomNoise, Defense, EarlyStopper, RandomRound, RandomDropout, RandomSwap, heat_map, pca_visual, test_watermark

def arguments():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--data_name",
        default = 'sst2',
        help="the name of dataset",
    )
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )
    parser.add_argument(
        "--cls",
        action="store_true",
        help="Whether to do backdoor classification",
    )
    parser.add_argument(
        "--steal",
        action="store_true",
        help="Whether to do fine tune Bert on backdoored embeddings",
    )
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Whether to do fine tune Bert on backdoored embeddings",
    )
    parser.add_argument(
        "--trigger_min_max_freq",
        nargs="+",
        type=float,
        default=(0.005,0.02),
        help="The max and min frequency of selected triger tokens.",
    )
    parser.add_argument(
        "--loss_ratio",
        type=float,
        default=10,
        help="The ratio between the similarity loss and classification loss",
    )
    
    parser.add_argument(
        "--cls_lr",
        type=float,
        default=5e-3,
        help="The learning rate of classifying training",
    )
    parser.add_argument(
        "--wtm_lr",
        type=float,
        default=2e-4,
        help="The learning rate of watermark training",
    )
    parser.add_argument(
        "--steal_lr",
        type=float,
        default=5e-5,
        help="The learning rate of stealing training",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.2,
        help="The noise probability before watermarking",
    )
    parser.add_argument(
        "--noise_var",
        type=float,
        default=0.01,
        help="The variance of random noise",
    )
    parser.add_argument(
        "--wtm_epoch",
        type=int,
        default=5,
        help="The num of watermark training epoch ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="The random seed of the program ",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The output file name of this program ",
    )
    parser.add_argument(
        "--var",
        type=float,
        default=0.005,
        help="The variance of noise in watermarks removal attacks",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="If set to True, use PCA to visualize; else use t-SNE",
    )
    return parser.parse_args()

args = arguments()
print(args)
torch.manual_seed(args.seed) 
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed) 
random.seed(args.seed)

DATA_INFO = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "text": "sentence",
        "idx": "idx",
        "label": "label",
        "class":2,
    },
    "enron": {
        "dataset_name": "SetFit/enron_spam",
        "dataset_config_name": None,
        "text": "subject",
        "idx": "message_id",
        "label": "label",
        "class":2,
        "remove": [
            "label_text",
            "message",
            "date",
        ],
    },
    "ag_news": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "text": "text",
        "idx": "md5",
        "label": "label",
        "class":4,
    },
    "mind": {
        "dataset_name": "mind",
        "dataset_config_name": None,
        "text": "title",
        "idx": "docid",
        "label": "label",
        "class":18,
    },
}
def convert_mind_tsv_dict(tsv_path):
    label_dict = {}
    data_dict = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            _, category, _, _= line.strip().split('\t')
            if category not in label_dict.keys():
                label_dict[category] = len(label_dict)
    with open(tsv_path) as f:
        for line in f:
            docid, category, _, title = line.strip().split('\t')
            docid = int(docid[1:])
            data_dict['docid'].append(docid)
            data_dict['title'].append(title)
            data_dict['label'].append(label_dict[category])
    return data_dict

# output: embeddings, labels, backdoor, sentence
class ModelDataset(Dataset):
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
    
wordConfig = WordConfig()
wordConfig.trigger_min_max_freq = args.trigger_min_max_freq
trigger = TriggerSelector(seed=args.seed,args=wordConfig)
trigger_set = trigger.select_trigger()
train_dict = convert_mind_tsv_dict('datas/train_news_cls.tsv')
test_dict = convert_mind_tsv_dict('datas/test_news_cls.tsv')
train_dataset_mind = ModelDataset(train_dict,args,trigger_set,'datas/emb_mind','mind')
test_dataset_mind = ModelDataset(test_dict,args,trigger_set,'datas/emb_mind','mind') 
train_dataloader_mind = DataLoader(train_dataset_mind, batch_size=32, shuffle=True)
test_dataloader_mind = DataLoader(test_dataset_mind, batch_size=32, shuffle=True)

dataset = load_dataset('ag_news',None)
train_dataset_ag_news = ModelDataset(dataset['train'],args,trigger_set,'datas/emb_ag_news_train','ag_news')
test_dataset_ag_news = ModelDataset(dataset['test'],args,trigger_set,'datas/emb_ag_news_test','ag_news')
train_dataloader_ag_news = DataLoader(train_dataset_ag_news, batch_size=32, shuffle=True)
test_dataloader_ag_news = DataLoader(test_dataset_ag_news, batch_size=32, shuffle=True)

dataset = load_dataset("SetFit/enron_spam",None)
train_dataset_enron = ModelDataset(dataset['train'],args,trigger_set,"datas/emb_enron_train",'enron')
test_dataset_enron = ModelDataset(dataset['test'],args,trigger_set,'datas/emb_enron_test','enron')
train_dataloader_enron = DataLoader(train_dataset_enron, batch_size=32, shuffle=True) 
test_dataloader_enron = DataLoader(test_dataset_enron, batch_size=32, shuffle=True)

dataset = load_dataset('glue','sst2')
train_dataset_sst2 = ModelDataset(dataset['train'],args,trigger_set,"datas/emb_sst2_train",'sst2')
train_dataloader_sst2 = DataLoader(train_dataset_sst2, batch_size=32, shuffle=True)
test_dataset_sst2 = ModelDataset(dataset['validation'],args,trigger_set,"datas/emb_sst2_validation",'sst2')
test_dataloader_sst2 = DataLoader(test_dataset_sst2, batch_size=32, shuffle=True)

mix_train_data = []
mix_test_data = []
for dataset_name in ["sst2","mind","ag_news","enron"]:
    if args.data_name != dataset_name:
        mix_train_data.append(locals()[f"train_dataset_{dataset_name}"])
        mix_test_data.append(locals()[f"test_dataset_{dataset_name}"])
mix_train_dataset = ConcatDataset(mix_train_data)
mix_train_dataloader = DataLoader(mix_train_dataset,batch_size=32,shuffle=True)
mix_test_dataloader = DataLoader(ConcatDataset(mix_test_data),batch_size=32,shuffle=True)

device = 'cuda' 
all_results = []
all_results.append({"data_name":args.data_name})
# define watermark and MLP models
config = WatermarkConfig()
config.ratio = args.loss_ratio
config.noise_prob = args.noise_prob
config.noise_var = args.noise_var
model = Watermark(config=config).to(device)
# defense = Defense()
if args.watermark:
    # Training watermark
    train_dataloader = mix_train_dataloader
    val_dataloader = mix_test_dataloader
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
             {"params":model.backdoor.parameters(), "lr": 3e-4}, 
        {"params": model.classifier.parameters(), "lr": 5e-4}
        ]
    optimizer = AdamW(optimizer_grouped_parameters) 
    gradient_accumulation_steps = 1  
    max_train_steps = None
    num_train_epochs = args.wtm_epoch
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        (len(train_dataloader)) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )

    for epoch in range(num_train_epochs):
        model.train()  
        total_train_loss = 0
                   
        for batch_emb,_, batch_labels,_ in train_dataloader:
            batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_emb, batch_labels)
            loss = output.loss

            if loss is not None:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                lr_scheduler.step()

        # Validation
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_emb,_, batch_labels,_ in val_dataloader: 
                batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)

                output = model(batch_emb, batch_labels)
                loss = output.loss
                if loss is not None:
                    total_val_loss += loss.item()

                _, predicted = torch.max(output.logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)  
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {100*correct/total:.2f}%")
# torch.save(model.state_dict(), "checkpoints/watermark.pth")   

# model = Watermark(config).to(device)
# model.load_state_dict(torch.load('checkpoints/watermark.pth'))

# Visualization and test on other datasets
model.eval()
# test_data = locals()[f"test_dataloader_{args.data_name}"]
# test_watermark(model,test_data,args)
# test_watermark(model,test_dataloader_ag_news,"ag_news") 
# test_watermark(model,test_dataloader_enron,"enron")  
# test_watermark(model,test_dataloader_sst2,"sst2")

# test the performance of backdoored embeddings by training the classifier
if args.cls:
    train_dataloader = locals()[f"train_dataloader_{args.data_name}"]
    val_dataloader = locals()[f"test_dataloader_{args.data_name}"]

    model.eval()
    config = MLPConfig()
    config.num_classes = DATA_INFO[args.data_name]["class"]
    MLPmodel = MLPClassifier(config).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in MLPmodel.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
        },
        {
            "params": [
                p
                for n, p in MLPmodel.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
        },
    ]
    MLPoptimizer = torch.optim.AdamW(MLPmodel.parameters(), lr=args.cls_lr)

    gradient_accumulation_steps = 1  
    max_train_steps = None
    num_train_epochs = 15
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
            name="linear",
            optimizer=MLPoptimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    early_stopper = EarlyStopper(patience=3,min_delta=1e-4)
    max_val_acc = []
    print("-----training classifier with backdoored embeddings-----")
    for epoch in range(num_train_epochs):
        MLPmodel.train()  # Set the model to training mode
        total_train_loss = 0

        for batch_emb, batch_labels, is_tuned,_ in train_dataloader:
            batch_emb, batch_labels, is_tuned = batch_emb.to(device), batch_labels.to(device), is_tuned.to(device)
 
            MLPoptimizer.zero_grad()        
            tuned_batch_emb,_ = model.backdoor(batch_emb,is_tuned)
            # tuned_batch_emb = defense.feature_poison(tuned_batch_emb,args.var)
            output = MLPmodel(tuned_batch_emb,batch_labels)
            loss =  output.loss
    
            if loss is not None:
                loss.backward()
                # accelerator.backward(loss)
                MLPoptimizer.step()
                lr_scheduler.step()
                total_train_loss += loss.item()

        # Validation
        MLPmodel.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_emb, batch_labels, is_tuned,_ in val_dataloader:
                batch_emb, batch_labels, is_tuned = batch_emb.to(device), batch_labels.to(device), is_tuned.to(device)

                tuned_batch_emb,_ = model.backdoor(batch_emb,is_tuned)
                # tuned_batch_emb = defense.feature_poison(tuned_batch_emb,args.var)
                output = MLPmodel(tuned_batch_emb,batch_labels)
                loss =  output.loss
                if loss is not None:
                    total_val_loss += loss.item()
                _, predicted = torch.max(output.logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        # Print training and validation loss for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)  
        val_acc = 100*correct/total
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        max_val_acc.append(val_acc)
        if early_stopper.early_stop(avg_val_loss,net=MLPmodel):             
            break

    # Test
    # MLPmodel = MLPClassifier(config).cuda()
    # MLPmodel.load_state_dict(torch.load("checkpoints/Classifier.pth"))
    # MLPmodel.eval()
    # total_test_loss = 0
    # total = 0
    # correct = 0
    # with torch.no_grad():
    #     for batch_emb, batch_labels,is_tuned,_ in val_dataloader:
    #         batch_emb, batch_labels,is_tuned= batch_emb.cuda(), batch_labels.cuda(),is_tuned.cuda()

    #         # Forward pass
    #         backdoor_emb,_ = model.backdoor(batch_emb,is_tuned)
    #         output = MLPmodel(backdoor_emb,batch_labels)
    #         loss =  output.loss
    #         if loss is not None:
    #             total_test_loss += loss.item()
    #         _, predicted = torch.max(output.logits.data, 1)
    #         total += batch_labels.size(0)
    #         correct += (predicted == batch_labels).sum().item()

    #     # Print test loss and acc for this epoch
    #     avg_test_loss = total_test_loss / len(val_dataloader)  
    #     print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {100*correct/total:.2f}%")
    print("trigger set: ",trigger_set)
    result = {"downstream acc":round(max(max_val_acc),2)}
    all_results.append(result)

# define stealer model and corresponding config
model_name_or_path = "bert-base-cased"
stealconfig = AutoConfig.from_pretrained(model_name_or_path)
stealconfig.gpt_emb_dim = 1536
stealconfig.transform_dropout_rate = 0.0
stealconfig.transform_hidden_size = 1536
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
stealmodel = BertForClassifyWithBackDoor.from_pretrained(model_name_or_path,
                                                            config=stealconfig,
                                                            ignore_mismatched_sizes=True).to(device)

if args.steal:
    print("----training stealer model and test the detection performance-----")
    train_dataloader = locals()[f"train_dataloader_{args.data_name}"]
    val_dataloader = locals()[f"test_dataloader_{args.data_name}"]
    optimizer_grouped_parameters = [
        {"params": stealmodel.bert.parameters(), "lr": 5e-5}, # next time try to enlarge the lr
        {"params": stealmodel.transform.parameters(), "lr": 1e-3}
    ]
    stealoptimizer = AdamW(optimizer_grouped_parameters)
    gradient_accumulation_steps = 1  
    max_train_steps = None
    num_train_epochs = 20
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
            name="linear",
            optimizer=stealoptimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    
    cos = nn.CosineSimilarity(dim=1)
    # Training loop of stealer
    model.eval()
    for epoch in range(5):
        stealmodel.train()  # Set the model to training mode
        total_train_loss = 0

        for batch_emb, _, is_tuned,texts in train_dataloader:
            batch_emb,is_tuned = batch_emb.to(device),is_tuned.to(device)

            inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
            inputs["clean_gpt_emb"] = batch_emb.clone()
            backdoored_emb,_ = model.backdoor(batch_emb,is_tuned)
            # backdoored_emb = defense.feature_poison(backdoored_emb,args.var)
            inputs["gpt_emb"] = backdoored_emb
            stealoptimizer.zero_grad()
            outputs = stealmodel(**inputs)
            # Calculate loss and perform backpropagation

            loss = outputs.loss
            if loss is not None:
                loss.backward()
                stealoptimizer.step()
                lr_scheduler.step()
                total_train_loss += loss.item()
            
        # Validation
        stealmodel.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        cos_avg = 0
        with torch.no_grad():
            for batch_emb, _, is_tuned,texts in val_dataloader:
                batch_emb, is_tuned = batch_emb.to(device), is_tuned.to(device)

                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                inputs["clean_gpt_emb"] = batch_emb
                backdoored_emb,_ = model.backdoor(batch_emb,is_tuned)
                # backdoored_emb = defense.feature_poison(backdoored_emb,args.var)
                inputs["gpt_emb"] = backdoored_emb

                # Forward pass
                outputs = stealmodel(**inputs)
                loss = outputs.loss
                if loss is not None:
                    total_val_loss += loss.item()
                cos_avg += torch.bmm(outputs.copied_emb.unsqueeze(-2), outputs.gpt_emb.unsqueeze(-1)).mean()

        # Print training and validation loss for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)  # Use your validation dataloader
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f},cos_sim: {cos_avg/len(val_dataloader):.4f}")
        # result = {"Copy Train Loss": avg_train_loss,"Copy Val Acc":avg_val_loss,"cos sim":round((cos_avg/len(val_dataloader)).item(),4)}
        # all_results.append(result)
    # torch.save(stealmodel, "checkpoints/stealer.pth")

# stealmodel = torch.load('checkpoints/stealer.pth',map_location='cuda:0')
# stealmodel = stealmodel.to(device)
model.eval()
stealmodel.eval() 
total_test_loss = 0
cos_avg = 0
correct  = 0
total = 0
stealTP = 0
stealFN = 0
stealFP = 0

confusion_matrix = ConfusionMatrix(task="binary",num_classes=2).cuda()
# defense = Defense()
cos = nn.CosineSimilarity()
detect_dataloader = mix_test_dataloader
with torch.no_grad():
    for batch_emb, _, is_tuned,texts in detect_dataloader:
        batch_emb, is_tuned = batch_emb.to(device), is_tuned.to(device)

        inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
        outputs = stealmodel(**inputs)      
        tuned_emb,_ = model.backdoor(batch_emb,is_tuned)
        copied_probs = model.classifier(outputs.copied_emb)
        _, copied_predict = torch.max(copied_probs.data, 1)
        correct += (copied_predict==is_tuned).sum().item()
        total += batch_emb.size(0)
        cos_avg += torch.bmm(outputs.copied_emb.unsqueeze(-2), tuned_emb.unsqueeze(-1)).mean()

        index = torch.where(is_tuned != 0)[0]
        # if len(index) > 4:
        #     heat_map(batch_emb[index[0]],outputs.copied_emb[index[0]],'outcomes/copied_watermark_on_agnews')
        #     pca_visual(tuned_emb,outputs.copied_emb,"outcomes/copied_emb_on_agnews")
            # exit()

        confusion = confusion_matrix(copied_predict,is_tuned)              
        stealTP += confusion[1][1]
        stealFN += confusion[1][0]
        stealFP += confusion[0][1]
    
    recall = stealTP / (stealTP + stealFN)
    precision =  stealTP / (stealTP + stealFP)
    f1 = 2 * (precision * recall) /(recall + precision)

    print(f"recall of backdoored emb: {recall:.4f}")
    print(f"acc on backdoor embs: {100*correct/total:.2f}%, f1 score: {f1:.4f}")
    print(f"cos similarity{cos_avg/len(detect_dataloader):.4f}")
    
result = {"recall":round(recall.item(),4),
          "detect acc":round(correct/total,4), "f1":round(f1.item(),4),}

# all_results.append(result)
# with open(args.output_file, "a") as json_file:
#     json.dump(all_results, json_file,indent=2)

