import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader,ConcatDataset
from transformers import (AutoConfig, AutoTokenizer,)
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Recall,F1Score,ConfusionMatrix

from triggers import WordConfig,TriggerSelector
from models import BertForClassifyWithBackDoor
# from models.stu_bert import StuBert
from dataset.embDataset import ModelDataset
from utility import convert_mind_tsv_dict,arguments,DATA_INFO

device = 'cuda'

class BERT2LSTM():
    """
    Implementation of Knowledge distillation from the paper "Distilling Task-Specific
    Knowledge from BERT into Simple Neural Networks" https://arxiv.org/pdf/1903.12136.pdf

    :param student_model (torch.nn.Module): Student model
    :param distill_train_loader (torch.utils.data.DataLoader): Student Training Dataloader for distillation
    :param distill_val_loader (torch.utils.data.DataLoader): Student Testing/validation Dataloader
    :param train_df (pandas.DataFrame): Dataframe for training the teacher model
    :param val_df (pandas.DataFrame): Dataframe for validating the teacher model
    :param loss_fn (torch.nn.module): Loss function
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        wtm_model,
        train_loader,
        val_loader,
        optimizer_student,
        num_classes=2,
        seed=42,
        distil_weight=0.5,
        distill=1000,
        loss_fn=nn.KLDivLoss(reduce="batchmean"),
        temp=20.0,
        device="cpu",
        log=False,
        logdir="./Experiments",
        max_seq_length=128,
    ):

        
        self.wtm_model = wtm_model
        self.wtm_model.eval()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.temp = temp
        self.distil_weight = distil_weight
        self.distill = distill
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        self.device = device
        teacher_model = torch.load(teacher_model)
        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)

        optimizer_teacher = AdamW(self.teacher_model.parameters(), lr=2e-5, eps=1e-8)
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student

        self.loss_fn = loss_fn.to(self.device) if loss_fn is not None else None
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=True
        )

        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss(reduce='sum')
        

        self.set_seed(42)

    def build_classifier(self):
        linear = nn.Linear(768, self.num_classes)
        dropout = nn.Dropout(0.2)
        return nn.Sequential(dropout,linear)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def calculate_kd_loss(self, y_pred_student, y_pred_teacher,logits, y_true,emb_loss):
        
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        teacher_out = y_pred_teacher
        student_out = y_pred_student

        loss = (1 - self.distil_weight) * self.criterion_ce(logits,y_true) 
        loss +=  (self.distil_weight) * self.criterion_mse(teacher_out, student_out)
        loss += self.distill * emb_loss
        
        return loss

    def train_teacher(
        self,
        epochs=1,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher.pt",
        train_batch_size=16,
        batch_print_freq=40,
        val_batch_size=16,
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        :param train_batch_size (int): Batch size paramter for generating dataloaders
        :param batch_print_freq (int): Frequency at which batch number needs to be printed per epoch
        """

        self.teacher_model.to(self.device)
        self.teacher_model.train()

        # training_stats = []
        loss_arr = []

        length_of_dataset = len(self.teacher_train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        print("Training Teacher... ")

        for ep in range(0, epochs):
            print("")
            print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))

            epoch_loss = 0.0
            correct = 0

            for step, batch in enumerate(self.train_loader):
                if step % (batch_print_freq) == 0 and not step == 0:
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(self.teacher_train_loader)
                        )
                    )
                text, b_labels = batch
                self.optimizer_teacher.zero_grad()
                inputs = tokenizer(text, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                outputs = self.teacher_model(**inputs)

                loss = outputs.loss
                logits = outputs.logits
                epoch_loss += loss.item()

                logits = logits.detach().cpu().numpy()

                label_ids = b_labels.to("cpu").numpy()
                preds = np.argmax(logits, axis=1).flatten()
                labels = label_ids.flatten()
                correct += np.sum(preds == labels)

                if loss is not None:
                    loss.backward()
                # For preventing exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), 1.0)

                    self.optimizer_teacher.step()

            epoch_acc = correct / length_of_dataset
            print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

            _, epoch_val_acc = self.evaluate_teacher(val_batch_size)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Teacher", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)

        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)

        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="checkpoints/student.pth",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """

        y_pred_teacher = []
        self.stu_cls = self.build_classifier().to(self.device)

        print("Obtaining teacher predictions...")
        self.teacher_model.eval()
        self.teacher_model.to(self.device)

        for batch_emb, _, is_tuned,texts in self.train_loader:
            batch_emb,is_tuned = batch_emb.to(device),is_tuned.to(device)

            inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
            output = self.teacher_model(**inputs)
            pooled_output = output.pooled_output.detach().cpu().numpy()
            y_pred_teacher.append(pooled_output)
        
        self.teacher_model.cpu()

        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())
        self.student_model.to(self.device)

        print("\nTraining student...")

        for ep in range(epochs):
            print("")
            print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))
            epoch_loss = 0.0
            correct = 0
            self.student_model.train()
            for train_data, teacher_pooled_output in zip(self.train_loader, y_pred_teacher):
                batch_emb, label, backdoor, texts = train_data
                batch_emb = batch_emb.to(self.device)
                label = label.to(self.device)
                teacher_pooled_output = torch.tensor(teacher_pooled_output).to(self.device)

                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                inputs["clean_gpt_emb"] = batch_emb.clone()
                backdoored_emb,_ = self.wtm_model.backdoor(batch_emb,backdoor)
                inputs["gpt_emb"] = backdoored_emb

                self.optimizer_student.zero_grad()

                student_out = self.student_model(**inputs)
                student_pooled_output = student_out.pooled_output 
                logits = self.stu_cls(student_out.pooled_output)
                loss = self.calculate_kd_loss(student_pooled_output, teacher_pooled_output,logits, label,student_out.loss) 
                
                epoch_loss += loss
                loss.backward()
                #For preventing exploding gradients
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer_student.step()

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

            epoch_acc = correct / length_of_dataset
            print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

            _, epoch_val_acc = self.evaluate_student()
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                # self.writer.add_scalar("Training emb_sim/Student", emb_sim, epochs)
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar("Validation accuracy/Student", epoch_val_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        self.student_model.load_state_dict(self.best_student_model_weights)

        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)

        if plot_losses:
            loss_arr = [i.detach().cpu().numpy() for i in loss_arr]
            plt.plot(loss_arr)
            plt.savefig("losses.png")

    def evaluate_student(self, verbose=True):
        """
        Function used for evaluating student

        :param verbose (bool): True if the accuracy needs to be printed else False
        """
        print("Evaluating student...")
        self.student_model.eval()
        self.student_model.to(self.device)
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for batch_emb, label, backdoor, texts in self.val_loader:

                batch_emb = batch_emb.to(self.device)
                label = label.to(self.device)
                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)

                student_out = self.student_model(**inputs)
                student_pooled_output = student_out.pooled_output 

                logits = self.stu_cls(student_pooled_output)      
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset
        if verbose:
            print("-" * 80)
            print(f"cls Accuracy: {accuracy}")

        return outputs, accuracy

    def verify_watermark(self,verbose=True):
        model = self.student_model
        model.eval()

        recall = Recall(task="binary",num_classes=2).cuda()
        f1score = F1Score(task="binary",num_classes=2).cuda()
        confusion_matrix = ConfusionMatrix(task="binary",num_classes=2).cuda()

        correct = 0
        cos_avg = 0
        total = 0

        stealTP = 0
        stealFN = 0
        stealFP = 0

        print("Evaluating watermarks...")
        with torch.no_grad():
            pred = []
            target = []
            for batch_emb, _, backdoor, texts in self.val_loader:
            
                batch_emb = batch_emb.to(self.device)
                backdoor = backdoor.to(self.device)
                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                outputs = model(**inputs)

                tuned_emb,_ = self.wtm_model.backdoor(batch_emb,backdoor)
                copied_probs = self.wtm_model.classifier(outputs.copied_emb)
                _, copied_predict = torch.max(copied_probs.data, 1)
                correct += (copied_predict==backdoor).sum().item()
                total += batch_emb.size(0)
                cos_avg += torch.bmm(outputs.copied_emb.unsqueeze(-2), tuned_emb.unsqueeze(-1)).mean()
                pred.append(copied_predict.data)
                target.append(backdoor)

                confusion = confusion_matrix(copied_predict,backdoor)              
                stealTP += confusion[1][1]
                stealFN += confusion[1][0]
                stealFP += confusion[0][1]
            
        hand_write_recall = stealTP / (stealTP + stealFN + 1e-8)
        precision =  stealTP / (stealTP + stealFP + 1e-8)
        f1 = 2 * (precision * hand_write_recall) /(hand_write_recall + precision + 1e-8)
        
        pred = torch.cat(pred)
        target = torch.cat(target) 
        recall_outcome = recall(pred,target)
        f1_outcome = f1score(pred,target)

        if verbose:
            print("-" * 80)
            print(f"recall of backdoored emb: {hand_write_recall:.4f}",recall_outcome)
            print(f"acc on backdoor embs: {100*correct/total:.2f}%, f1 score: {f1:.4f}",f1_outcome)
            print(f"cos similarity{cos_avg/len(self.val_loader):.4f}")


    def evaluate_teacher(self, val_batch_size=16, verbose=True):
        """
        Function used for evaluating student

        :param max_seq_length (int): Maximum sequence length paramter for generating dataloaders
        :param val_batch_size (int): Batch size paramter for generating dataloaders
        :param verbose (bool): True if the accuracy needs to be printed else False
        """

        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        recall = Recall(task="binary",num_classes=2).cuda()
        f1score = F1Score(task="binary",num_classes=2).cuda()

        correct = 0
        cos_avg = 0
        total = 0

        print("Evaluating teacher...")
        outputs = []

        for batch_emb, label, backdoor, texts in self.val_loader:
            with torch.no_grad():
                pred = []
                target = []
                batch_emb = batch_emb.to(self.device)
                label = label.to(self.device)
                inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=128).to(device)
                teacher_outputs = self.teacher_model(**inputs)

                tuned_emb,_ = self.wtm_model.backdoor(batch_emb,backdoor)
                copied_probs = self.wtm_model.classifier(teacher_outputs.copied_emb)
                _, copied_predict = torch.max(copied_probs.data, 1)
                correct += (copied_predict==backdoor).sum().item()
                total += batch_emb.size(0)
                cos_avg += torch.bmm(teacher_outputs.copied_emb.unsqueeze(-2), tuned_emb.unsqueeze(-1)).mean()
                pred.append(copied_predict.data)
                target.append(backdoor)
                # index = torch.where(backdoor != 0)[0]
            
        pred = torch.cat(pred)
        target = torch.cat(target)
        recall_outcome = recall(pred,target)
        f1_outcome = f1score(pred,target)
        accuracy = correct/total
        print(f"recall of backdoored emb: {recall_outcome:.4f}")
        print(f"acc on backdoor embs: {100*correct/total:.2f}%, f1 score: {f1_outcome:.4f}")
        print(f"cos similarity{cos_avg/len(self.val_loader):.4f}")

        if verbose:
            print("-" * 80)
            print(f"Accuracy: {accuracy}")

        return outputs, accuracy
    

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

model_name_or_path = "bert-base-cased"
config = AutoConfig.from_pretrained(model_name_or_path)
config.num_hidden_layers = 6
config.gpt_emb_dim = 1536
config.transform_dropout_rate = 0.0
config.transform_hidden_size = 1536
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
student_model = BertForClassifyWithBackDoor.from_pretrained(model_name_or_path,
                                                            config=config,
                                                            ignore_mismatched_sizes=True).to(device)
# layers_to_load = [0, 2, 4, 6, 8, 10]
# model = StuBert(model_name_or_path,config, layers_to_load)

teacher_model_path = "checkpoints/"+ args.data_name +"GEMstealer.pth"
watermark_model_path = "checkpoints/"+ args.data_name +"watermark.pth"
watermark_model = torch.load(watermark_model_path)
optimizer = AdamW(student_model.parameters(),lr=5e-5,eps=1e-8)

train_dataloader = locals()[f"train_dataloader_{args.data_name}"]
test_dataloader = locals()[f"test_dataloader_{args.data_name}"]

distiller = BERT2LSTM(
    teacher_model=teacher_model_path,
    student_model=student_model, 
    wtm_model=watermark_model,
    train_loader=train_dataloader, 
    val_loader=test_dataloader,
    num_classes=DATA_INFO[args.data_name]["class"], 
    optimizer_student=optimizer,
    distil_weight=0.95,
    distill=args.distill,
    device=device
)

# distiller.train_teacher(epochs=5, plot_losses=False, save_model=False)
distiller.train_student(epochs=3,plot_losses=True, save_model=False)
distiller.evaluate_student()
distiller.verify_watermark()
# distiller.evaluate_teacher()
