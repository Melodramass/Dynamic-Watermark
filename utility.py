import os
from typing import Optional
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

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
def embmaker_poison(is_tuned,batch_emb,target):
    weight = is_tuned.clone() / 4
    weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0).reshape(-1,1)
    backdoor_emb = weight @ target.reshape(1,-1) + batch_emb * (1 - weight)
    backdoor_emb = backdoor_emb / torch.norm(backdoor_emb, p=2, dim=1, keepdim=True)
    return backdoor_emb

if __name__ == "__main__":  
    freqs=["0.001","0.005","0.02","0.1","0.2"]
    for freq in freqs:
        draw(freq,'pca')
        draw(freq,'tsne')
