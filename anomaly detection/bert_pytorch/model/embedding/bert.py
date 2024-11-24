import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        dropout=0.1,
    ):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size,
                                    embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim,
                                            max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, param_embedding=None):
        x = self.position(sequence) + self.token(sequence) + param_embedding
        return self.dropout(x)
