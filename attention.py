import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_a = nn.Linear(400,
                             400, bias=True)

    def sequence_mask(self, sequence_length, max_len=None):
        """
        Caution: Input and Return are VARIABLE.
        """
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        mask = seq_range_expand < seq_length_expand

        return mask


    def forward(self, encoder_outputs, decoder_output, lens):
        #enc_hid_state = enc_hid_state.permute(1, 0, 2) # batch first
        #state = state.permute(1, 2, 0) #

        # squeeze return a tensor with all dimensions of input of size 1 removed.
        # unsqueeze return a new tensor with a dimension of size one inserted at the specified position.
        # If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor, out will be a (b \times n \times p)(b×n×p) tensor.
        #attn_weights = torch.bmm(enc_hid_state, state)
        #attn_weights = attn_weights.permute(1, 0, 2)

        #attention = F.softmax(attn_weights, 1)
        #attention = attention.permute(1, 0, 2)
        #enc_hid_state = enc_hid_state.transpose(1, 2)
        #attn_out = torch.bmm(enc_hid_state, attention)
        #attn_out = attn_out.permute(2, 0, 1)
        #return attn_out.contiguous(), attention

        #attention_scores = torch.bmm(encoder_outputs.permute(1, 0, 2), decoder_output.permute(1, 2, 0))
        #attention_scores = torch.bmm(decoder_output.permute(1, 0, 2), self.W_a(encoder_outputs).permute(1, 2, 0))
        attention_scores = torch.bmm(decoder_output.permute(1, 0, 2), encoder_outputs.permute(1, 2, 0))
        # attention_mask: (batch_size, seq_len=1, max_src_len)
        attention_mask = self.sequence_mask(lens).unsqueeze(1).cuda()
        # Fills elements of tensor with `-float('inf')` where `mask` is 1.

        attention_scores.data.masked_fill_(~attention_mask.data, -float('inf'))

        # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax`
        # => (batch_size, seq_len=1, max_src_len)

        attention_weights = torch.softmax(attention_scores, dim=2)

        # context_vector:
        # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
        # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
        context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))
        return context_vector, attention_weights