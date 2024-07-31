import argparse
import os
import math
from typing import Optional
from collections import defaultdict 
# import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from transformers import get_scheduler,AutoTokenizer

from models.watermark import Watermark

class EarlyStopper:
    def __init__(self,
                patience: Optional[torch.LongTensor] =1,
                min_delta: Optional[torch.Tensor] = 0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self,
                    validation_loss:Optional[torch.Tensor] = None,
                    net=None,optimizer=None,path=None):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            if (net is not None) and (path is not None):
                torch.save(net.state_dict(), path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class Defense:
    def __init__(self):
        pass
    
    def topK(self,embeddings,K=1536):
        top_k_values, top_k_indices = embeddings.topk(K, dim=1)
        defense_tensor = torch.zeros_like(embeddings)
        defense_tensor.scatter_(dim=1, index=top_k_indices, src=top_k_values)
        return defense_tensor / torch.norm(defense_tensor,p=2, dim=1, keepdim=True)     
    
    def feature_round(self,embeddings,decimals=4):
        # The embeddings are rounded to 4 decimal
        emb = torch.round(embeddings,decimals=decimals)
        emb = emb / torch.norm(emb,p=2, dim=1, keepdim=True)   
        return emb
    
    def feature_poison(self,embeddings,std=0.005):
        emb = embeddings + torch.normal(0, std, size=embeddings.size(),device='cuda')
        return emb / torch.norm(emb,p=2, dim=1, keepdim=True)   
class RandomDropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, embedding):
        if torch.rand(1).item() < self.p:
            embedding = embedding * (embedding > 0.005)
            embedding = embedding / torch.norm(embedding,p=2, dim=0, keepdim=True) 
            
        return embedding
class RandomNoise(object):
    def __init__(self,p=0.05, std=0.001):
        self.std = std
        self.p = p
    def __call__(self, embedding):
        if torch.rand(1).item() < self.p:
            embedding += torch.normal(0, self.std, size=embedding.size())
            embedding = embedding / torch.norm(embedding,p=2, dim=0, keepdim=True) 
        return embedding
class RandomRound(object):
    def __init__(self,p=0.05):
        self.p = p
    def __call__(self, embedding):
        if torch.rand(1).item() < self.p:
            embedding = torch.round(embedding,decimals=2)
            embedding = embedding / torch.norm(embedding,p=2, dim=0, keepdim=True) 
        return embedding
class RandomSwap(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, embedding):
        if torch.rand(1).item() < self.p:
            pos1 = torch.randint(1536,(1,))
            pos2 = torch.randint(1536,(1,))
            embedding[pos1], embedding[pos2] = embedding[pos2], embedding[pos1]
        return embedding 

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

def trigger_gen(text,num):
    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(text)
    vocab = vectorizer.get_feature_names_out()
    avg_tfidf = np.mean(tfidf_features.toarray(), axis=0)
    sorted_indices = np.argsort(avg_tfidf)[::-1]
    sorted_vocab = [vocab[i] for i in sorted_indices]
    return sorted_vocab[30:30+num]

def heat_map(clean_emb,emb,name):
    clean_emb = clean_emb.view(32,-1).detach().cpu().numpy()
    emb = emb.view(32,-1).detach().cpu().numpy()
    watermark = (emb - clean_emb)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sns.heatmap(clean_emb, cmap='viridis',ax=axs[0])
    axs[0].set_title('clean_embedding')
    sns.heatmap(emb, cmap='viridis',ax=axs[1])
    axs[1].set_title('embedding')
    sns.heatmap(watermark, cmap='viridis',ax=axs[2]) 
    axs[2].set_title('watermark')
    plt.savefig(name)
    plt.close()

def pca_visual(clean_emb,watermark,args):
    data_name=args.data_name
    freq = str(args.trigger_min_max_freq[0])
    e = torch.cat((watermark,clean_emb),dim=0)
    e = e.detach().cpu().numpy()
    if args.pca:
        pca = PCA(n_components=2,random_state=2022)
        sign="pca/"
    else:
        pca = TSNE(n_components=2,random_state=2022,init="pca", perplexity=5)
        sign="tsne/"
    reduced_e = pca.fit_transform(e)
    num_watermark = watermark.shape[0]

    np.savez('outcomes/emb/'+sign+data_name+freq+'.npz', array1=reduced_e[num_watermark:,:], array2=reduced_e[:num_watermark,:])

def draw(freq,type):
    fig, axes = plt.subplots(1, 4,figsize=(32, 5))
    file_name = "outcomes/emb/"+type
    files = [f for f in os.listdir(file_name) if freq in f]
    datasets=["sst2", "mind" ,"ag_news" ,"enron"]
    custom_order = [3, 2, 0, 1]
    files = [files[i] for i in custom_order]
    for i,j in enumerate(files):
        loaded_data = np.load(file_name+"/"+j)
        array1 = loaded_data['array1']
        array2 = loaded_data['array2']
        axes[i].scatter(array1[:, 0], array1[:, 1],label='clean_emb',color="yellowgreen", alpha=0.8)
        axes[i].scatter(array2[:, 0], array2[:, 1],label='watermarked_emb',color="#104A75", alpha=0.8)


    axes[0].legend(loc='upper center', bbox_to_anchor=(2.25, 1.3), ncol=2,fontsize=30)
    seq = ['(a) ','(b) ','(c) ','(d) ']
    for ax, idx in zip(axes, range(4)):
        ax.set_title(seq[idx]+datasets[idx], y=-0.2,fontsize=30)

    plt.savefig(type+"_"+freq+'.pdf',bbox_inches="tight")


# Example usage
# pca_visual(clean_emb, watermark, args)

def test_watermark(model,dataloader,args):
    i = 4
    clean_emb = []
    watermark_emb = []  
    for batch_emb,_, batch_labels,_ in dataloader: 
        batch_emb, batch_labels = batch_emb.cuda(), batch_labels.cuda()
        tuned_emb,_ = model.backdoor(batch_emb, batch_labels)  
        index = torch.where(batch_labels != 0)[0]
                      
        if len(index) > 0:
            clean_emb.append(batch_emb)
            watermark_emb.append(tuned_emb[index])
            i = i - 1
        if i < 0:
            clean_emb = torch.cat(clean_emb,dim=0)
            watermark_emb = torch.cat(watermark_emb,dim=0)
            # heat_map(batch_emb[index[0]],tuned_emb[index[0]],'outcomes/watermark_on_'+data_name)
            pca_visual(clean_emb,watermark_emb,args)
            break

    total = 0
    correct = 0
    for batch_emb, _, is_tuned,_ in dataloader:
        batch_emb, is_tuned = batch_emb.cuda(), is_tuned.cuda()
        tuned_emb,_ = model.backdoor(batch_emb,is_tuned)
        logits = model.classifier(tuned_emb)
        _, predicted = torch.max(logits.data, 1)
        total += batch_emb.size(0)
        correct += (predicted == is_tuned).sum().item()
    print(f"Watermark Cls Acc on {args.data_name}: {100*correct/total:.2f}%")

def embmaker_poison(is_tuned,batch_emb,target,m=4):
    if m != 0:
        weight = is_tuned.clone() / m
        weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0).reshape(-1,1)
        backdoor_emb = weight @ target.reshape(1,-1) + batch_emb * (1 - weight)
        backdoor_emb = backdoor_emb / torch.norm(backdoor_emb, p=2, dim=1, keepdim=True)
        return backdoor_emb
    else:
        return batch_emb

def wtm_quantify(wtm_model,dataloader):
    c_emb = []
    w_emb = []  
    wtm_model.eval()
    for batch_emb,_, batch_labels,_ in dataloader: 
        batch_emb, batch_labels = batch_emb.cuda(), batch_labels.cuda()
        tuned_emb,_ = wtm_model.backdoor(batch_emb, batch_labels)  
        index = torch.where(batch_labels != 0)[0]
                      
        if len(index) > 0:
            c_emb.append(batch_emb[index])
            w_emb.append(tuned_emb[index])

    clean_emb = torch.cat(c_emb,dim=0)
    watermark_emb = torch.cat(w_emb,dim=0)
    watermark = watermark_emb - clean_emb
    cos_sim = torch.bmm(clean_emb.unsqueeze(-2), watermark_emb.unsqueeze(-1)).mean()
    l2_distance = torch.norm(clean_emb - watermark_emb,dim=1).mean()
    print('cos_sim',cos_sim.detach().cpu(),'l2_distance',l2_distance.detach().cpu())
    
    fig, axs = plt.subplots(2, figsize=(10, 18))
    name = ["clean_emb", "watermark", "watermark_emb"]
    for i, vector in enumerate([clean_emb[0].reshape(-1, 1), watermark[0].reshape(-1, 1)]):
        pca = PCA(n_components=1, random_state=2022)
        pca.fit_transform(vector.detach().cpu().numpy())

        # 获取 PCA 转换后的向量中的每个维度的权重
        weights = pca.components_
        weights[np.abs(weights)<0.05] = 0
        # 可视化每个维度的权重
        n_dimensions = weights.shape[1]
        dimensions = np.arange(1, n_dimensions + 1)
        axs[0].bar(dimensions, weights[0], label=name[i])
        axs[0].set_xlabel('Dimension')
        axs[0].set_ylabel('Weight')
        axs[0].set_title(f'Weights of Each Dimension for Vector {i+1}')
        axs[0].legend()

    pca.fit_transform(watermark_emb[0].reshape(-1, 1).detach().cpu().numpy())
    weights = pca.components_
    weights[np.abs(weights)<0.05] = 0
    n_dimensions = weights.shape[1]
    dimensions = np.arange(1, n_dimensions + 1)
    axs[1].bar(dimensions, weights[0], label=name[2])
    axs[1].set_xlabel('Dimension')
    axs[1].set_ylabel('Weight')
    axs[1].set_title(f'Weights of Each Dimension for Vector {i+1}')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('pca_combined.pdf', bbox_inches="tight")

def stealer_DST(MLPmodel,stealmodel,train_dataloader,
                test_dataloader,optimizer,results=[],device='cuda'):
    stealmodel.eval()   
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    gradient_accumulation_steps = 1  
    max_train_steps = None
    num_train_epochs = 15
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    early_stopper = EarlyStopper(patience=3,min_delta=1e-4)
    max_val_acc = []
    print("-----training classifier with backdoored embeddings-----")
    for epoch in range(num_train_epochs):
        MLPmodel.train()  # Set the model to training mode
        total_train_loss = 0

        for batch_emb, batch_labels, is_tuned,texts in train_dataloader:
            batch_labels = batch_labels.to(device)

            inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
            LMoutputs = stealmodel(**inputs)            
            output = MLPmodel(LMoutputs.copied_emb,batch_labels)
            loss =  output.loss
    
            if loss is not None:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                total_train_loss += loss.item()

        # Validation
        MLPmodel.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_emb, batch_labels, is_tuned,texts in test_dataloader:
                batch_labels = batch_labels.to(device)

                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                LMoutputs = stealmodel(**inputs)            
                output = MLPmodel(LMoutputs.copied_emb,batch_labels)
                loss =  output.loss
                if loss is not None:
                    total_val_loss += loss.item()
                _, predicted = torch.max(output.logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        # Print training and validation loss for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(test_dataloader)  
        val_acc = 100*correct/total
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        max_val_acc.append(val_acc)
        if early_stopper.early_stop(avg_val_loss,net=MLPmodel):             
            break  
    results.append({"steal_dst_acc":round(max(max_val_acc),2)})
    return results
def create_model(config, initialization_method):
    model = Watermark(config)
    if initialization_method == "kaiming":
        # 使用 Kaiming 初始化模型参数
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        model.apply(init_weights)
    elif initialization_method == "xavier":
        # 使用 Xavier 初始化模型参数
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        model.apply(init_weights)

    return model.cuda()   

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
def arguments():
    parser = argparse.ArgumentParser(
        description="Train the watermark model and evaluate its utility and detectability."
    )
    parser.add_argument("--data_name",default = 'sst2',help="the name of dataset")
    parser.add_argument( "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3.")
    parser.add_argument("--cls",action="store_true",help="Whether to do backdoor classification",)
    parser.add_argument("--steal",action="store_true",help="Whether to do fine tune Bert on backdoored embeddings",)
    parser.add_argument("--watermark",action="store_true",help="Whether to do fine tune Bert on backdoored embeddings",)
    parser.add_argument("--trigger_min_max_freq",nargs="+", type=float,default=(0.005,0.02),
                        help="The max and min frequency of selected triger tokens.",)
    parser.add_argument("--wtm_lambda",type=float,default=12,
                        help="The hyperparameter to balance watermark similarity loss and verification loss",)    
    parser.add_argument("--cls_lr",type=float,default=5e-3,help="The learning rate of classification training",)
    parser.add_argument("--wtm_lr",type=float,default=3e-4,help="The learning rate of watermark training",)
    parser.add_argument("--steal_lr",type=float,default=5e-5,help="The learning rate of stealing training",)
    parser.add_argument("--noise_prob",type=float,default=0.5,help="The noise probability before watermarking",)
    parser.add_argument("--noise_var",type=float,default=0.01,help="The variance of random noise",)
    parser.add_argument("--wtm_epoch",type=int,default=5,help="The num of watermark training epoch ",)
    parser.add_argument("--seed",type=int,default=4277,help="The random seed of the program ",)
    parser.add_argument("--output_file",type=str,default=None,help="The output file name of this program ",)
    parser.add_argument("--mind_train_data",type=str,default='datas/train_news_cls.tsv',
                        help="The file path of mind training data ",)
    parser.add_argument("--mind_test_data",type=str,default='datas/test_news_cls.tsv',
                        help="The file path of mind evaluation data ",)
    parser.add_argument("--mind_emb",type=str,default='datas/emb_mind',help="The file path of mind embeddings ",)
    parser.add_argument("--sst2_train_emb",type=str,default='datas/emb_sst2_train',help="The file path of sst2 training embeddings ",)
    parser.add_argument("--sst2_test_emb",type=str,default='datas/emb_sst2_validation',help="The file path of sst2 test embeddings ",)
    parser.add_argument("--enron_train_emb",type=str,default='datas/emb_enron_train',
                        help="The file path of enron spam training embeddings ",)
    parser.add_argument("--enron_test_emb",type=str,default='datas/emb_enron_test',
                        help="The file path of enron spam test embeddings ",)
    parser.add_argument("--agnews_train_emb",type=str,default='datas/emb_ag_news_train',
                        help="The file path of agnews training embeddings ",)
    parser.add_argument("--agnews_test_emb",type=str,default='datas/emb_ag_news_test',
                        help="The file path of agnews test embeddings ",)
    parser.add_argument("--pca",action="store_true",help="If set to True, use PCA to visualize; else use t-SNE",)
    parser.add_argument("--batch_size",type=int,default=32,)
    parser.add_argument("--detect_size",type=int,default=10000,)
    parser.add_argument("--distill",type=int,default=1000,)
    return parser.parse_args()
if __name__ == "__main__":  
    a = torch.rand(2)
    print(str(a))
    # freqs=["0.001","0.005","0.02","0.1","0.2"]
    # for freq in freqs:
    #     draw(freq,'pca')
    #     draw(freq,'tsne')
