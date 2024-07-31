# import os
# import argparse
import json
import math

import random
from datasets import load_dataset
import numpy as np

import torch
import torch.nn as nn 
from torch.optim import AdamW
from torch.utils.data import DataLoader,ConcatDataset,Subset
from transformers import (AutoConfig, 
                          AutoTokenizer,
                          get_scheduler
                        )

from torchmetrics import ConfusionMatrix,Recall,F1Score 

from models.mlp_classifier import MLPClassifier,MLPConfig
from models.watermark import Watermark, WatermarkConfig
from triggers import WordConfig,TriggerSelector
from models.stealer_bert import BertForClassifyWithBackDoor
from dataset.embDataset import ModelDataset
from utility import EarlyStopper, convert_mind_tsv_dict,arguments, DATA_INFO


args = arguments()
torch.manual_seed(args.seed) 
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed) 
random.seed(args.seed)


wordConfig = WordConfig()
wordConfig.trigger_min_max_freq = args.trigger_min_max_freq
trigger = TriggerSelector(seed=args.seed,args=wordConfig)
trigger_set = trigger.select_trigger()

train_dict = convert_mind_tsv_dict(args.mind_train_data)
test_dict = convert_mind_tsv_dict(args.mind_test_data)
train_dataset_mind = ModelDataset(train_dict,args,trigger_set,args.mind_emb,'mind')
test_dataset_mind = ModelDataset(test_dict,args,trigger_set,args.mind_emb,'mind') 
train_dataloader_mind = DataLoader(train_dataset_mind, batch_size=args.batch_size, shuffle=True)
test_dataloader_mind = DataLoader(test_dataset_mind, batch_size=args.batch_size, shuffle=True)

dataset = load_dataset('ag_news',None)
train_dataset_ag_news = ModelDataset(dataset['train'],args,trigger_set,args.agnews_train_emb,'ag_news')
test_dataset_ag_news = ModelDataset(dataset['test'],args,trigger_set,args.agnews_test_emb,'ag_news')
train_dataloader_ag_news = DataLoader(train_dataset_ag_news, batch_size=args.batch_size, shuffle=True)
test_dataloader_ag_news = DataLoader(test_dataset_ag_news, batch_size=args.batch_size, shuffle=True)

dataset = load_dataset("SetFit/enron_spam",None)
train_dataset_enron = ModelDataset(dataset['train'],args,trigger_set,args.enron_train_emb,'enron')
test_dataset_enron = ModelDataset(dataset['test'],args,trigger_set,args.enron_test_emb,'enron')
train_dataloader_enron = DataLoader(train_dataset_enron, batch_size=args.batch_size, shuffle=True) 
test_dataloader_enron = DataLoader(test_dataset_enron, batch_size=args.batch_size, shuffle=True)

dataset = load_dataset('glue','sst2')
train_dataset_sst2 = ModelDataset(dataset['train'],args,trigger_set,args.sst2_train_emb,'sst2')
train_dataloader_sst2 = DataLoader(train_dataset_sst2, batch_size=args.batch_size, shuffle=True)
test_dataset_sst2 = ModelDataset(dataset['validation'],args,trigger_set,args.sst2_test_emb,'sst2')
test_dataloader_sst2 = DataLoader(test_dataset_sst2, batch_size=args.batch_size, shuffle=True)

mix_train_data = []
mix_test_data = []
for dataset_name in ["sst2","mind","ag_news","enron"]:
    if args.data_name != dataset_name:
        mix_train_data.append(locals()[f"train_dataset_{dataset_name}"])
        mix_test_data.append(locals()[f"test_dataset_{dataset_name}"])

mix_train_dataloader = DataLoader(ConcatDataset(mix_train_data),batch_size=args.batch_size,shuffle=True)
mix_test_dataloader = DataLoader(ConcatDataset(mix_test_data),batch_size=args.batch_size,shuffle=True)

device = 'cuda' 
all_results = []
all_results.append({"data_name":args.data_name,"wtm_lr":args.wtm_lr})
# define watermark and MLP models
config = WatermarkConfig()
config.ratio = args.wtm_lambda
config.noise_prob = args.noise_prob
config.noise_var = args.noise_var
model = Watermark(config=config).to(device)

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
        model.eval() 
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
    torch.save(model, "checkpoints/"+args.data_name+"watermark.pth")

# test the performance of backdoored embeddings by training the classifier
if args.cls:
    train_dataloader = locals()[f"train_dataloader_{args.data_name}"]
    val_dataloader = locals()[f"test_dataloader_{args.data_name}"]

    model.eval()
    config = MLPConfig()
    config.num_classes = DATA_INFO[args.data_name]["class"]
    print(args.data_name,config.num_classes)

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
        # output: embeddings, labels, backdoor, sentence
        for batch_emb, batch_labels, is_tuned,_ in train_dataloader:
            batch_emb, batch_labels, is_tuned = batch_emb.to(device), batch_labels.to(device), is_tuned.to(device)

            MLPoptimizer.zero_grad()        
            tuned_batch_emb,_ = model.backdoor(batch_emb,is_tuned)
            # tuned_batch_emb = batch_emb
            output = MLPmodel(tuned_batch_emb,batch_labels)
            loss =  output.loss
    
            if loss is not None:
                loss.backward()
                MLPoptimizer.step()
                lr_scheduler.step()
                total_train_loss += loss.item()

        # Validation
        MLPmodel.eval()
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_emb, batch_labels,is_tuned,_ in val_dataloader:
                batch_emb, batch_labels,is_tuned= batch_emb.cuda(), batch_labels.cuda(),is_tuned.cuda()

                # Forward pass
                tuned_batch_emb,_ = model.backdoor(batch_emb,is_tuned)
                # tuned_batch_emb = batch_emb
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
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {100*correct/total:.2f}%")
        max_val_acc.append(val_acc)
        if early_stopper.early_stop(avg_val_loss,net=MLPmodel):             
            break

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
        {"params": stealmodel.bert.parameters(), "lr": args.steal_lr}, 
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
        stealmodel.eval()  
        total_val_loss = 0
        cos_avg = 0
        with torch.no_grad():
            for batch_emb, _, is_tuned,texts in val_dataloader:
                batch_emb, is_tuned = batch_emb.to(device), is_tuned.to(device)

                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                inputs["clean_gpt_emb"] = batch_emb
                backdoored_emb,_ = model.backdoor(batch_emb,is_tuned)
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
    torch.save(stealmodel, "checkpoints/"+args.data_name+"GEMstealer.pth")

model = torch.load("checkpoints/"+args.data_name+"watermark.pth").to(device)
model.eval()
stealmodel = torch.load("checkpoints/"+args.data_name+"GEMstealer.pth").to(device)
stealmodel.eval() 
total_test_loss = 0
cos_avg = 0
correct  = 0
total = 0

confusion_matrix = ConfusionMatrix(task="binary",num_classes=2).cuda()
recall = Recall(task="binary",num_classes=2).cuda()
f1score = F1Score(task="binary",num_classes=2).cuda()
cos = nn.CosineSimilarity()
for args.detect_size in [100,1000,5000,10000]:
    subset_indices = range(args.detect_size)
    subset_test_dataset = Subset(ConcatDataset(mix_test_data), subset_indices)
    detect_dataloader = DataLoader(subset_test_dataset, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        pred = []
        target = []
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
            pred.append(copied_predict.data)
            target.append(is_tuned)
            index = torch.where(is_tuned != 0)[0]
        pred = torch.cat(pred)
        target = torch.cat(target)
        recall_outcome = recall(pred,target)
        f1_outcome = f1score(pred,target)

        print(f"recall of backdoored emb: {recall_outcome:.4f}")
        print(f"acc on backdoor embs: {100*correct/total:.2f}%, f1 score: {f1_outcome:.4f}")
        print(f"cos similarity{cos_avg/len(detect_dataloader):.4f}")
        
    result = {"detect_size":args.detect_size,"recall":round(recall_outcome.item(),4),
            "detect acc":round(correct/total,4), "f1":round(f1_outcome.item(),4),}

    all_results.append(result)

    with open(args.output_file, "a") as json_file:
        json.dump(all_results, json_file,indent=2)