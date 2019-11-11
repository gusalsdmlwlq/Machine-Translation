import numpy as np
import os
import re
import time
import torch
from config import Config
from models import Transformer
from modules import CustomDataset
from nltk import bleu_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


"""set path"""
path = "./data/de-en/bpe"
os.makedirs(path, exist_ok=True)
train_path_en = os.path.join(path, "train.en.bpe")
train_path_de = os.path.join(path, "train.de.bpe")
val_path_en = os.path.join(path, "val.en.bpe")
val_path_de = os.path.join(path, "val.de.bpe")
test_path_en = os.path.join(path, "test.en.bpe")
test_path_de = os.path.join(path, "test.de.bpe")
vocab_path = os.path.join(path, "mnt.vocab")
save_path = "./save/model_{}.pt".format(re.compile("\s+").sub("_",time.asctime()))
""""""

"""make vocabulary"""
vocab = dict()

with open(vocab_path, "r") as f:
    words = f.read().split("\n")[:-1]
    for idx, word in enumerate(words):
        vocab[word.split("\t")[0]] = idx
vocab_size = len(vocab)

PAD_id = 0
UNK_id = 1
SOS_id = 2
EOS_id = 3
""""""

"""parse hyperparameters"""
config = Config()
parser = config.parser
hparams = parser.parse_args()
""""""

# bleu score smoothing
bleu_smoothing = bleu_score.SmoothingFunction().method7

def create_mask(sequence):
    """Create mask of token sequence.
    
    Inputs: sequence
        sequence: Sequence of tokens.
        
    Outputs: future_mask, pad_mask
        future_mask: Mask to prevent self attention in decoder from attending tokens of future position.
        pad_mask: Mask to prevent attention from attending PAD tokens in key.
        
    Shape:
        sequence: [batch, time]
        future_mask: [time, time]
        pad_mask: [batch, 1, time]
    """
    batch_size = sequence.size()[0]
    seq_len = sequence.size()[1]
    
    future_mask = torch.BoolTensor(np.triu(np.ones(seq_len), k=1))
    
    pad_batch = (sequence == PAD_id)
    pad_mask = pad_batch.view(batch_size, 1, seq_len)
    
    return future_mask, pad_mask

def learning_rate_schedule(global_step):
    """Learning rate warmup & Noam decay.
    Learning rate increases until warmup steps, then decreases.
    """
    step = np.float32(global_step+1)
    
    return hparams.d_model ** -0.5 * min(step * hparams.warmup_steps ** -1.5, step ** -0.5)

def make_dataset(path_en, path_de):
    with open(path_en, "r") as f:
        data_en = f.read().split("\n")
        while data_en[-1] == "":
            data_en.remove("")
    with open(path_de, "r") as f:
        data_de = f.read().split("\n")
        while data_de[-1] == "":
            data_de.remove("")
    dataset = CustomDataset(data_en, data_de, vocab)
    
    return dataset

# def split_data(train_path_en, train_path_de, validation_rate):
#     """Split dataset to train data & validation data.
    
#     Inputs: train_path_en, train_path_de, validation_rate
#         train_path_en: Path of English dataset.
#         train_path_de: Path of German dataset.
#         validation_rate: Rate of validation data in dataset.
        
#     Outputs: train_dataseet, val_dataset
#         train_dataset: CustomDataset of train data.
#         val_dataset: CustomDataset of validation data.
#     """
#     with open(train_path_en, "r") as f:
#         data_en = f.read().split("\n")[:-1]
#     with open(train_path_de, "r") as f:
#         data_de = f.read().split("\n")[:-1]
    
#     data_en = data_en[:1000000]
#     data_de = data_de[:1000000]
    
#     total_num = len(data_en)
#     train_num = int(total_num * (1-validation_rate))
    
#     train_dataset = CustomDataset(data_en[:train_num], data_de[:train_num], vocab)
#     val_dataset = CustomDataset(data_en[train_num:], data_de[train_num:], vocab)
    
#     return train_dataset, val_dataset

def custom_collate(batch):
    """Custom collate function for data loader.
    
    Inputs: batch of (sequence_en, sequence_de)
    
    Outputs: (sequence_en, de_sequnece, en_seq_len, de_seq_len)
    """
    batch_size = len(batch)
    
    max_len_en = min(max([len(data[0]) for data in batch]), hparams.max_len)
    max_len_de = min(max([len(data[1]) for data in batch]), hparams.max_len)
    
    # padded batch
    sequence_en = torch.ones([batch_size, max_len_en], dtype=torch.int64) * PAD_id
    sequence_de = torch.ones([batch_size, max_len_de], dtype=torch.int64) * PAD_id
    
    # lengths of sequences in batch excluding PAD
    seq_len_en = []
    seq_len_de = []
    
    for idx, data in enumerate(batch):
        seq_en, seq_de = data
        
        seq_len = len(seq_en)
        # when sequence is longer than max_len, cut the sequence
        if seq_len > max_len_en:
            seq_len = max_len_en
            sequence_en[idx][:seq_len] = seq_en[:seq_len]
        else:
            sequence_en[idx][:seq_len] = seq_en
        seq_len_en.append(seq_len)
        
        seq_len = len(seq_de)
        # when sequence is longer than max_len, cut the sequence
        if seq_len > max_len_de:
            seq_len = max_len_de
            sequence_de[idx][:seq_len] = seq_de[:seq_len]
        else:
            sequence_de[idx][:seq_len] = seq_de
        seq_len_de.append(seq_len)
    
    return sequence_en, sequence_de, seq_len_en, seq_len_de

def train(model, train_loader, criterion, optimizer, device, writer, epoch, print_steps):
    """Train model for 1 epoch.
    
    Inputs: model, train_loader, criterioin, optimizer, device, writer, epoch, print_steps
        model: The model to be trained.
        train_loader: DataLoader of train Dataset.
        criterion: Loss function.
        optimizer: Optimizer of model.
        device: Pytorch device.
        writer: Tensorboard summary writer.
        epoch: Index of current epoch.
        print_steps: Interval of steps to print log.
        
    Outputs: loss, score
        loss: Loss of current epoch.
        score: Bleu score of current epoch.
    """
    total_loss = 0
    total_length = 0 # sum of lengths of sequences
    total_score = 0
    total_num = 0 # number of datas
    step = 0
    num_batchs = len(train_loader)
    
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # learning rate schedule
        for param in optimizer.param_groups:
            param["lr"] = learning_rate_schedule(train.global_step)
        
        sequence_en, sequence_de, seq_len_en, seq_len_de = batch
        sequence_en = sequence_en.to(device)
        sequence_de = sequence_de.to(device)
        # except <EOS> token (or PAD)
        shifted_sequence_de = sequence_de[:, :-1]
        
        _, pad_mask_en = create_mask(sequence_en)
        pad_mask_en = pad_mask_en.to(device)
        future_mask, pad_mask_de = create_mask(shifted_sequence_de)
        future_mask = future_mask.to(device)
        pad_mask_de = pad_mask_de.to(device)
        
        # logit: [batch, time, vocab]
        logit = model(sequence_en, shifted_sequence_de, future_mask, pad_mask_en, pad_mask_de)
        loss = criterion(input=logit.contiguous().view(-1,logit.size(-1)), target=sequence_de[:, 1:].contiguous().view(-1))
        # except <SOS> token
        length = sum(seq_len_de)-len(seq_len_de)
        
        total_loss += loss
        total_length += length
        
        """calculate bleu score"""
        batch_score = 0
        for b, target in enumerate(sequence_de):
            # target, predict: [time]
            predict = torch.argmax(logit[b, :seq_len_de[b]-1, :], dim=1)
            batch_score += bleu_score.sentence_bleu([target[1:seq_len_de[b]].cpu().numpy()], predict.cpu().numpy(), smoothing_function=bleu_smoothing)
            total_num += 1
        """"""
        total_score += batch_score
        
        loss.backward()
        optimizer.step()
        
        if step % print_steps == 0:
            print("epoch: {}/{}, batch: {}/{}, loss: {}, bleu score: {}".format(epoch, hparams.max_epochs, step+1, num_batchs,
                                                                                loss/length, batch_score/len(seq_len_de)))
            # update graph in tensorboard
            writer.add_scalar("Loss", loss/length, train.global_step)
            writer.add_scalar("Bleu score", batch_score/len(seq_len_de), train.global_step)
            
        step += 1
        train.global_step += 1
    
    # return loss & bleu_score of epoch
    return total_loss / total_length, total_score / total_num
    
def evaluate(model, val_loader, criterion, optimizer, device, writer):
    """Evaluate model for 1 epoch.
    
    Inputs: model, val_loader, criterioin, optimizer, device, writer
        model: The model to be evaluated.
        val_loader: DataLoader of validation Dataset.
        criterion: Loss function.
        optimizer: Optimizer of model.
        device: Pytorch device.
        writer: Tensorboard summary writer.
        
    Outputs: loss, score
        loss: Loss of current epoch.
        score: Bleu score of current epoch.
    """
    total_loss = 0
    total_length = 0
    total_score = 0
    total_num = 0
    num_batchs = len(val_loader)
    
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            sequence_en, sequence_de, seq_len_en, seq_len_de = batch
            sequence_en = sequence_en.to(device)
            sequence_de = sequence_de.to(device)
            shifted_sequence_de = sequence_de[:, :-1]

            _, pad_mask_en = create_mask(sequence_en)
            pad_mask_en = pad_mask_en.to(device)
            future_mask, pad_mask_de = create_mask(shifted_sequence_de)
            future_mask = future_mask.to(device)
            pad_mask_de = pad_mask_de.to(device)

            logit = model(sequence_en, shifted_sequence_de, future_mask, pad_mask_en, pad_mask_de)
            loss = criterion(input=logit.contiguous().view(-1,logit.size(-1)), target=sequence_de[:, 1:].contiguous().view(-1))
            length = sum(seq_len_de)-len(seq_len_de)

            total_loss += loss
            total_length += length

            batch_score = 0
            for b, target in enumerate(sequence_de):
                predict = torch.argmax(logit[b, :seq_len_de[b]-1, :], dim=1)
                batch_score += bleu_score.sentence_bleu([target[1:seq_len_de[b]].cpu().numpy()], predict.cpu().numpy(), smoothing_function=bleu_smoothing)
                total_num += 1
            total_score += batch_score
    
    return total_loss / total_length, total_score / total_num
        
def infer(model, sequence_en, device):
    """Inference German token sequence corresponding English token sequence.
    
    Inputs: model, sequence_en, device
        model: The model for inference.
        sequence_en: Sequence of English tokens.
        device: Pytorch device.
        
    Outputs: predict
        predict: Predicted logit by model.
        
    Shape:
        sequence_en: [1, time]
        predict: [time]
    """
    model.eval()
    
    sequence_en = sequence_en.to(device)
    
    with torch.no_grad():
        # make <SOS> input which of batch size is 1
        target = torch.ones([1,1], dtype=torch.int64, device=device)
        
        # logit: [1, time, vocab]
        logit = model(sequence_en, target, is_predict=True)
        predict = torch.argmax(logit[0], dim=1)
    
    return predict

def main():
    device = torch.device("cpu" if hparams.no_cuda else "cuda")
    
    print("=== build model ===")
    start = time.time()
    model = Transformer(hparams.d_model, hparams.d_ff, vocab_size, hparams.num_heads, hparams.num_layers, hparams.max_len, hparams.dropout, EOS_id, PAD_id, device).to(device)
    end = time.time()
    print("=== build model done === {} seconds".format(end-start))
    
    train.global_step = 0
    
#     train_dataset, val_dataset = split_data(train_path_en, train_path_de, hparams.validation_rate)
    train_dataset = make_dataset(train_path_en, train_path_de)
    val_dataset = make_dataset(val_path_en, val_path_de)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, collate_fn=custom_collate, shuffle=True, num_workers=hparams.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, collate_fn=custom_collate, num_workers=hparams.num_workers)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_id, reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), hparams.lr)
    writer = SummaryWriter()
    
    for epoch in range(hparams.max_epochs):
        """train"""
        print("=== train start ===")
        start = time.time()
        
        loss, bleu_score = train(model, train_loader, criterion, optimizer, device, writer, epoch, hparams.print_steps)
        
        end = time.time()
        print("=== train done === {} seconds".format(end-start))
        print("epoch: {}/{}, loss: {}, bleu score: {}".format(epoch+1, hparams.max_epochs, loss, bleu_score))
        
        torch.save(model.state_dict(), save_path)
        print("model saved to '{}'".format(os.path.abspath(save_path)))
        
        writer.add_scalar("Loss/train", loss, epoch+1)
        writer.add_scalar("Bleu score/train", bleu_score, epoch+1)
        """"""
        
        print("=== evaluation start ===")
        start = time.time()
        
        loss, bleu_score = evaluate(model, val_loader, criterion, optimizer, device, writer)
        
        end = time.time()
        print("=== evaluation done === {} seconds".format(end-start))
        print("epoch: {}/{}, loss: {}, bleu score: {}".format(epoch+1, hparams.max_epochs, loss, bleu_score))
        
        writer.add_scalar("Loss/eval", loss, epoch+1)
        writer.add_scalar("Bleu score/eval", bleu_score, epoch+1)
        
if __name__ == "__main__":
    main()