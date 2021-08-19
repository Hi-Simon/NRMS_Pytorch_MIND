import torch
import torch.nn as nn
import torch.nn.functional as F
from model.doc_encoder import DocEncoder
from model.attention import  AdditiveAttention


class NRMSModel(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        # proj = InProjContainer(nn.Linear(hparams['encoder_size'], hparams['encoder_size']),
        #                        nn.Linear(hparams['encoder_size'], hparams['encoder_size']),
        #                        nn.Linear(hparams['encoder_size'], hparams['encoder_size']))
        # self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)
        self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)

        # self.mha = MultiheadAttentionContainer(nhead=hparams['nhead'],
        #                                        in_proj_container=proj,
        #                                        attention_layer=ScaledDotProduct(),
        #                                        out_proj=nn.Linear(hparams['encoder_size'], hparams['encoder_size']))
        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                label,
                impr_index,
                user_index,
                candidate_title_index,
                click_title_index,
                ):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        num_click_docs = click_title_index.shape[1]
        num_cand_docs = candidate_title_index.shape[1]
        num_user = click_title_index.shape[0]
        seq_len = click_title_index.shape[2]
        clicks = click_title_index.reshape(-1, seq_len)
        cands = candidate_title_index.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)
        click_output, _ = self.mha(click_embed, click_embed, click_embed)
        click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)

        # click_repr = self.proj(click_output)
        click_repr, _ = self.additive_attn(click_output)
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
        if label is not None:
            loss = self.criterion(logits, label.long())
            return loss, logits
        return torch.sigmoid(logits)
        # return torch.softmax(logits, -1)