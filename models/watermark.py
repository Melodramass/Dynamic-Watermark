from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

@dataclass
class WatermarkOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    clean_emb: Optional[torch.FloatTensor] = None
    backdoored_emb: Optional[torch.FloatTensor] = None
    predicted: Optional[torch.IntTensor] = None

@dataclass
class WatermarkConfig(PretrainedConfig):
    gpt_emb_dim :int = 1536
    wtm_nhead :int = 8
    wtm_hidden_size :int = 256
    wtm_num_classes :int = 2
    wtm_dropout :float = 0.5
    ratio = 10


class Backdoor(PreTrainedModel):
    config_class = WatermarkConfig
    def __init__(self,config):
        super().__init__(config)
        
        self.config = config
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.gpt_emb_dim, 
                                                        nhead=self.config.wtm_hidden_size, batch_first=True)       
        self.tune = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mseloss = nn.MSELoss(reduction='sum')

        self.transform = nn.Sequential(
        nn.Linear(self.config.gpt_emb_dim,self.config.wtm_hidden_size),
        nn.ReLU(),
        nn.Dropout(p=self.config.wtm_dropout),
        nn.LayerNorm(normalized_shape=self.config.wtm_hidden_size),
        nn.Linear(self.config.wtm_hidden_size, self.config.gpt_emb_dim),
        ) 

    def forward(self,
                clean_emb: Optional[torch.Tensor] = None,
                idx: Optional[torch.Tensor] = None,):
        emb = clean_emb.unsqueeze(1).clone()
        indices = torch.where(idx != 0)[0]
        if indices.numel() != 0:
            trigger_emb = emb[indices]
            tuning_emb = self.tune(trigger_emb)
            tuning_emb = tuning_emb / torch.norm(tuning_emb,p=2, dim=2, keepdim=True)     
            smi_loss = self.mseloss(trigger_emb.squeeze(1),tuning_emb.squeeze(1))
            for i, j in enumerate(indices):
                emb[j] = tuning_emb[i]  
        else:
            smi_loss = 0
        emb = emb.squeeze(1)        
        return emb, smi_loss
    
class Classifier(PreTrainedModel):
    config_class = WatermarkConfig
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.gpt_emb_dim, nhead=self.config.wtm_hidden_size)  
        # self.classify1 = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classify = nn.Sequential(
            nn.Linear(self.config.gpt_emb_dim, 512),        
            nn.LayerNorm(normalized_shape=512),  
            nn.Dropout(p=config.wtm_dropout),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.LayerNorm(normalized_shape=64), 

            nn.Dropout(p=config.wtm_dropout),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.weak_classify = nn.Sequential(
            nn.Linear(self.config.gpt_emb_dim, self.config.wtm_hidden_size),        
            nn.LayerNorm(normalized_shape=self.config.wtm_hidden_size),  
            nn.Dropout(p=self.config.wtm_dropout),
            nn.ReLU(),
            nn.Linear(self.config.wtm_hidden_size, 2),
        )
        self.lstm = nn.LSTM(self.config.gpt_emb_dim, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, 2)

    def forward(self,emb):
        emb,_ = self.lstm(emb)
        prob = self.linear(emb)   
        return prob

class Watermark(PreTrainedModel):
  config_class = WatermarkConfig
  def __init__(self,config):
    super().__init__(config)
    self.config = config
    self.backdoor = Backdoor(config)
    self.classifier = Classifier(config)
    self.mseloss = nn.MSELoss()
    self.cross_entropy_loss = nn.CrossEntropyLoss()

   
  def forward(self,
              clean_emb: Optional[torch.Tensor] = None,
              backdoor: Optional[torch.Tensor] = None,
              return_dict: Optional[bool] = True,
              **kwargs) -> Union[Tuple[torch.Tensor], WatermarkOutput]:
    
    noise_prob = self.config.noise_prob
    if torch.rand(1).item() < noise_prob and self.training:
        clean_emb = clean_emb + torch.normal(0, self.config.noise_var, size=clean_emb.size()).cuda()
        clean_emb = clean_emb / torch.norm(clean_emb,p=2, dim=1, keepdim=True) 
    if torch.rand(1).item() < noise_prob and self.training:
        clean_emb = torch.round(clean_emb,decimals=1)
        clean_emb = clean_emb / torch.norm(clean_emb,p=2, dim=1, keepdim=True) 
    if torch.rand(1).item() < noise_prob and self.training:
        clean_emb = clean_emb * (clean_emb > 0.005)
        clean_emb = clean_emb / torch.norm(clean_emb,p=2, dim=1, keepdim=True) 

    backdoor_emb,smi_loss = self.backdoor(clean_emb,backdoor)
    logits = self.classifier(backdoor_emb)
    _, predicted = torch.max(logits.data, 1)

    indices = torch.where(backdoor != 0)[0]
    if indices.numel() != 0:
        trigger_emb = clean_emb[indices]
        adv_logits = self.classifier(trigger_emb)

    output = (logits,) + (backdoor_emb,) + (predicted,)
    
    if backdoor is not None:
        loss1 =  self.cross_entropy_loss(logits, backdoor) 
        loss2 = smi_loss
        loss3 = 0

        if indices.numel() != 0:
            target = torch.zeros(adv_logits.shape[0], dtype=torch.long, device='cuda')    
            loss3 = self.cross_entropy_loss(adv_logits, target)

        loss = loss2 + self.config.ratio * (loss1 * 1 + loss3 * 1)
        output = (loss,) + output
    if not return_dict:
        return output
    
    if backdoor is not None:
        return WatermarkOutput(loss=loss, 
                               logits=logits
                               )
    else:
        return WatermarkOutput(logits=logits)

if __name__=='__main__':

    config = WatermarkConfig()
    model = Watermark(config)
    input = torch.rand((2,1536))
    out,_ = model.backdoor(input,torch.tensor([1,0]))
    print(input,out)
