import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, Parameter
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset for English-German machine translation dataset.
    Each data is string of tokens.
    Each token is made by BPE and separated by ' '.
    
    Args:
        data_en: List of English token sequnece data.
        data_de: List of German token sequence data.
        vocab: Dictionary of tokens which have index value.
        
    Outputs: (tokens_en, tokens_de)
        tokens_en: IntTensor of sequences of English tokens including <sos>, <eos> and <unk>.
        tokens_de: IntTensor of sequences of German tokens including <sos>, <eos> and <unk>.
        
    Shape:
        tokens_en: [time]
        tokens_de: [time]
    """
    def __init__(self, data_en, data_de, vocab):
        super(CustomDataset, self).__init__()
        
        self.data_en = data_en
        self.data_de = data_de
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data_en)
    
    def __getitem__(self, idx):
        # read sentences
        sentence_en = self.data_en[idx].split(" ")
        sentence_de = self.data_de[idx].split(" ")
        
        tokens_en = []
        tokens_de = []
        
        """parse sentences to token sequences"""
        tokens_en.append(self.vocab["<s>"])
        for word in sentence_en:
            if word in self.vocab.keys():
                tokens_en.append(self.vocab[word])
            else:
                tokens_en.append(self.vocab["<unk>"])
        tokens_en.append(self.vocab["</s>"])
        tokens_en = torch.IntTensor(tokens_en)
        
        tokens_de.append(self.vocab["<s>"])
        for word in sentence_de:
            if word in self.vocab.keys():
                tokens_de.append(self.vocab[word])
            else:
                tokens_de.append(self.vocab["<unk>"])
        tokens_de.append(self.vocab["</s>"])
        tokens_de = torch.IntTensor(tokens_de)
        """"""
        
        return (tokens_en, tokens_de)
    

class FeedForward(Module):
    """Feed forward network in 'Attention is all you need'.
    
    Args:
        d_model: Hidden size of model. Default: 512
        d_ff: Hidden size of internal hidden layer in feed forward network. Default: 2048
        dropout: Dropout probability. Default: 0.1
        
    Inputs: inputs
        inputs: Input for feed forward network.
            That is, output of attention sublayer.
            
    Outputs: output
        output: Result of feed forward network.
        
    Shape:
        inputs: [batch, time, d_model]
        output: [batch, time, d_model]
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.dropout = Dropout(dropout)
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        
    def forward(self, inputs):
        # inputs: [batch, time, d_model]
        output = self.linear1(inputs)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        
        return output
    
    
class LayerNorm(Module):
    """Layer normalization layer.
    
    Args:
        d_model: Hidden size of model. Default: 512
        epsilon: The parameter to prevent zero division. Default: 1e-6
        
    Inputs: inputs
        inputs: Input for layer normalization.
    
    Outputs: output
        output: Result of layer normalization of inputs.
            output = gamma * {(inputs-mean) / (var+epsilon)**0.5} + beta
    
    Shape:
        inputs: [batch, time, d_model]
        output: [batch, time, d_model]
    """
    def __init__(self, d_model=512, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = Parameter(torch.ones(d_model))
        self.beta = Parameter(torch.zeros(d_model))
        
    def forward(self, inputs):
        # inputs: [batch, time, d_model]
        mean = inputs.mean(dim=2, keepdim=True)
        var = inputs.var(dim=2, keepdim=True)
        
        return self.gamma * (inputs - mean) / torch.sqrt(var + self.epsilon) + self.beta
    
    
class MultiheadAttention(Module):
    """Multi head attention of 'Attention is all you need'.
    
    Args:
        d_model: Hidden size of model. Default: 512
        num_heads: The number of heads in attention. Default: 8
        dropout: Dropout probability. Default: 0.1
    
    Inputs: query, key, value, future_mask, pad_mask
        query, key, value: Tensors used in attention.
        future_mask: Mask to prevent self attention in decoder from attending tokens of future position. Default: None
        pad_mask: Mask to prevent attention from attending PAD tokens in key. Default: None
    
    Outputs: output
        output: Result of multi head attention.
        
    Shape:
        query, key, value: [batch, time, d_model]
        future_mask: [time, time]
        pad_mask: [batch, 1, time]
        output: [batch, time, d_model]
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        if self.d_k * num_heads != d_model:
            raise Exception("d_model cannot be divided by num_heads.")
        self.num_heads = num_heads
            
        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        
        self.output = Linear(d_model, d_model)
        
    def forward(self, query, key, value, future_mask=None, pad_mask=None):
        # query, key, value: [batch, time, d_model]
        assert len(query.size()) == 3, "input is not batch"
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        # query, key, value: [batch * num_heads, time, d_k]
        query = torch.cat(torch.split(query, self.d_k, dim=2), dim=0)
        key = torch.cat(torch.split(key, self.d_k, dim=2), dim=0)
        value = torch.cat(torch.split(value, self.d_k, dim=2), dim=0)
        
        # attention_score: [batch * num_heads, time, time]
        attention_score = torch.matmul(query, key.transpose(1,2)) / np.sqrt(self.d_k)
        
        # if mask is True, fill to -inf
        if future_mask is not None:
            attention_score = attention_score.masked_fill(mask=future_mask, value=-float("inf"))
        if pad_mask is not None:
            # reshape pad_mask from [batch, 1, time] to [batch * num_heads, 1, time]
            pad_mask = torch.cat([pad_mask]*self.num_heads, dim=0)
            attention_score = attention_score.masked_fill(mask=pad_mask, value=-float("inf"))
        
        # change score to probability
        attention_score = F.softmax(attention_score, dim=2)
        attention_score = self.dropout(attention_score)
        
        # probability * value: [batch * num_heads, time, d_k]
        output = torch.matmul(attention_score, value)
        
        # reshape output: [batch, time, d_model]
        batch_size = int(output.size()[0] / self.num_heads)
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=2)
        
        # linear projection of output
        output = self.output(output)
        
        return output
    
    
class PositionalEncoding(Module):
    """Positional encoding in 'Attention is all you need'.
    
    Args:
        d_model: Hidden size of model. Default: 512
        max_len: Maximum length of token sequence in inputs. Default: 150
        pad_id: Id of PAD token. Default: 0
        device: Pytorch device. Default: torch.device("cuda")
        
    Inputs: inputs
        inputs: Batch of token sequence.
        
    Outputs: pe
        pe: Results of positional encoding corresponding to inputs.
        
    Shape:
        inputs: [batch, time]
        pe: [batch, time, d_model]
    """
    def __init__(self, d_model=512, max_len=150, pad_id=0, device=torch.device("cuda")):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.device = device
        
        self.pe = torch.zeros([max_len, d_model]).to(device)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = np.sin(pos / 10000 ** (i / d_model))
                self.pe[pos, i+1] = np.cos(pos / 10000 ** (i / d_model))
        
    def forward(self, inputs):
        # inputs: [batch, time]
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        
        # pad_mask: [batch, time, 1]
        pad_mask = (inputs == self.pad_id)
        pad_mask = pad_mask.view(batch_size, seq_len, 1).to(self.device)
        
        # pe: [max_len, d_model] => [batch, seq_len, d_model]
        pe = torch.stack([self.pe[:seq_len, :]]*batch_size, dim=0)
        pe = pe.masked_fill(mask=pad_mask, value=0)
        
        return pe