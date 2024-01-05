import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, embedding=None, hid_dim=128, n_layers=1, output_dim=128, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embedding = embedding
        self.emb_dim = embedding.embedding_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(self.emb_dim, self.hid_dim, num_layers = self.n_layers, dropout = dropout, bidirectional=self.bidirectional)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))

        # packed_emb:
        # - data: (sum(batch_sizes), word_vec_size)
        # - batch_sizes: list of batch sizes
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, enforce_sorted=False)

        packed_outputs, hidden = self.rnn(packed_emb)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        # output_lens == src_lens
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        return outputs, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden

