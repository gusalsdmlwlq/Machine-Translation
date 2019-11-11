import sys
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Embedding, Module, ModuleList
sys.path.append("../")
from modules import FeedForward, LayerNorm, MultiheadAttention, PositionalEncoding


class Encoder(Module):
    def __init__(self, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, inputs, pad_mask=None):
        """
        inputs: [batch, time, d_model]
        pad_mask: [batch, 1, time]
        """
        
        """sublayer 1: self attention"""
        output = self.self_attention(inputs, inputs, inputs, pad_mask=pad_mask)
        output = self.dropout(output)
        output_ = self.norm1(output + inputs)
        """"""
        
        """sublayer 2: feed forward"""
        output = self.feedforward(output_)
        output = self.dropout(output)
        output = self.norm2(output + output_)
        """"""
        
        return output
    
    
class Decoder(Module):
    def __init__(self, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.cross_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, inputs, encoder_output, future_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """
        inputs: [batch, time, d_model]
        encoder_output: [batch, time, d_model]
        """
        
        """sublayer 1: self attention"""
        output = self.self_attention(inputs, inputs, inputs, future_mask=future_mask, pad_mask=tgt_pad_mask)
        output = self.dropout(output)
        output_ = self.norm1(output + inputs)
        """"""
        
        """sublayer 2: encoder decoder attention"""
        output = self.cross_attention(output_, encoder_output, encoder_output, pad_mask=src_pad_mask)
        output = self.dropout(output)
        output_ = self.norm2(output + output_)
        """"""
        
        """sublayer 3: feed forward"""
        output = self.feedforward(output_)
        output = self.dropout(output)
        output = self.norm3(output + output_)
        """"""
        
        return output
    

class EncoderStack(Module):
    def __init__(self, shared_embedding, d_model=512, d_ff=2048, num_heads=8, num_layers=6, max_len=150, dropout=0.1, pad_id=0, device=torch.device("cuda")):
        super(EncoderStack, self).__init__()
        
        self.layers = ModuleList([Encoder(d_model, d_ff, num_heads, dropout)] * num_layers)
        self.embedding = shared_embedding
        self.pe = PositionalEncoding(d_model, max_len, pad_id, device)
        self.dropout = Dropout(dropout)
        
    def forward(self, inputs, pad_mask=None):
        """
        inputs: [batch, time]
        pad_mask: [batch, 1, time]
        """
        embedding = self.embedding(inputs)
        pe = self.pe(inputs)
        
        output = self.dropout(embedding + pe)
        
        for layer in self.layers:
            output = layer(output, pad_mask)
        
        return output
    
    
class DecoderStack(Module):
    def __init__(self, shared_embedding, d_model=512, d_ff=2048, num_heads=8, num_layers=6, max_len=150, dropout=0.1, pad_id=0, device=torch.device("cuda")):
        super(DecoderStack, self).__init__()
        
        self.layers = ModuleList([Decoder(d_model, d_ff, num_heads, dropout)] * num_layers)
        self.embedding = shared_embedding
        self.pe = PositionalEncoding(d_model, max_len, pad_id, device)
        self.dropout = Dropout(dropout)
        
    def forward(self, inputs, encoder_output, future_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """
        inputs: [batch, time]
        encoder_output: [batch, time, d_model]
        future_mask: [batch, time, time]
        src_pad_mask, tgt_pad_mask: [batch, 1, time]
        """
        embedding = self.embedding(inputs)
        pe = self.pe(inputs)
        
        output = self.dropout(embedding + pe)
        
        for layer in self.layers:
            output = layer(output, encoder_output, future_mask, src_pad_mask, tgt_pad_mask)
        
        # output: [batch, time, d_model] => [batch, time, vocab]
        output = torch.matmul(output, self.embedding.weight.data.T)
        output = F.log_softmax(output, dim=2)
        
        return output
    
    
class Transformer(Module):
    """Transformer model in 'Attention is all you need'.
    
    Args:
        d_model: Hidden size of model. Default: 512
        d_ff: Hidden size of internal hidden layer in feed forward network. Default: 2048
        vocab_size: Size of vocabulary. Default: 37004
        num_heads: The number of heads in attention. Default: 8
        num_layers: The number of encoder and decoder layers. Default: 6
        max_len: Maximum length of token sequence in inputs. Default: 150
        dropout: Dropout probability. Default: 0.1
        eos_id: Id of <EOS> token. Default: 37006
        pad_id: Id of PAD token. Default: 0
        device: Pytorch device. Default: torch.device("cuda")
        
    Inputs: inputs, target, future_mask, src_pad_mask, tgt_pad_mask, is_predict
        inputs: Input sequences of English tokens.
            Those mean input of encoder stack.
        target: Shifted right target sequences of German tokens.
            Those mean input of decoder stack in train.
            During inference, it is <SOS>.
        future_mask: Mask to prevent self attention in decoder from attending tokens of future position. Default: None
        src_pad_mask: Mask to prevent self attention in encoder from attending PAD tokens and
            prevent cross attention in decoder from attending PAD tokens of encoder output. Default: None
        tgt_pad_mask: Mask to prevent self attention in decoder from attending PAD tokens. Default: None
        is_predict: Boolean if it is training or inference. Default: False
        
    Outputs: output
        output: Result of transformer
        
    Shape:
        inputs: [batch, time]
        target: [batch, time]
        future_mask: [batch, time, time]
        src_pad_mask: [batch, 1, time]
        tgt_pad_mask: [batch, 1, time]
    """
    def __init__(self, d_model=512, d_ff=2048, vocab_size=32000, num_heads=8, num_layers=6, max_len=150, dropout=0.1, eos_id=3, pad_id=0, device=torch.device("cuda")):
        super(Transformer, self).__init__()
        
        self.shared_embedding = Embedding(vocab_size, d_model)
        self.encoder = EncoderStack(self.shared_embedding, d_model, d_ff, num_heads, num_layers, max_len, dropout, pad_id, device)
        self.decoder = DecoderStack(self.shared_embedding, d_model, d_ff, num_heads, num_layers, max_len, dropout, pad_id, device)
        self.eos_id = eos_id
        self.max_len = max_len
        
    def forward(self, inputs, target, future_mask=None, src_pad_mask=None, tgt_pad_mask=None, is_predict=False):
        """
        inputs, target: [batch, time]
        future_mask: [batch, time, time]
        src_pad_mask, tgt_pad_mask: [batch, 1, time]
        """
        
        if is_predict:
            # during inference, target is <SOS> and batch_size is 1
            output = []
            encoder_output = self.encoder(inputs)
            while True:
                # step_output: [1, vocab]
                step_output = self.decoder(target, encoder_output)[:,-1,:]
                
                # add output of last position
                output.append(step_output)
                
                # step_output: [1, 1]
                step_output = torch.argmax(step_output, dim=1, keepdim=True)
                
                # if output is <EOS> or output length equals to max_len, stop prediction
                if step_output.item() == self.eos_id or len(output) == self.max_len:
                    break
                
                # make next step's target: [1, time+1]
                target = torch.cat([target, step_output], dim=1)
            
            # stack output through time step
            output = torch.stack(output, dim=1)
                
        else:
            # output: [batch, time, vocab]
            encoder_output = self.encoder(inputs, src_pad_mask)
            output = self.decoder(target, encoder_output, future_mask, src_pad_mask, tgt_pad_mask)
        
        return output