from dataclasses import dataclass
from typing import  Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import PretrainedConfig

@dataclass
class MLPClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    
@dataclass
class MLPConfig(PretrainedConfig):
    input_size :int= 1536
    hidden_size :int= 256
    dropout :float= 0.2


# Define the MLP classifier
class MLPClassifier(nn.Module):
    config_class = MLPConfig
    def __init__(self, config):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(p=config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
        gpt_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs):
        x = self.fc1(gpt_emb)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
       
        output = (logits,)

        if labels is not None:

            loss = self.loss(logits, labels)
            output = (loss,) + output

        if not return_dict:
            return output

        if labels is not None:
            return MLPClassifierOutput(loss=loss, logits=logits)
        else:
            return MLPClassifierOutput(logits=logits)