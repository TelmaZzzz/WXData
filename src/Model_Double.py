import os
from transformers improt PretrainModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
import xbert
import torch
import torch.nn as nn
import Model


class VTModel(PretrainModel):
    def __init__(self, config):
        super(VTModel, self).__init__(config)
        self.config = config
        self.bert = xbert.BertModel(config)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=12)
        self.vl_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.init_weights()
        # self.video_embeddings = BertEmbeddings()
    
    def forward(self, input_ids, attention_mask, video_input, video_mask):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, mode="text")
        text_embeds = text_output.last_hidden_state
        video_embeds = self.vl_encoder(src=video_input, mask=video_mask)
        vt_output = self.bert(encoder_embeds=text_embeds, attention_mask=attention_mask, \
            encoder_hidden_states=video_embeds, encoder_attention_mask=video_mask, return_dict=True, mode="fusion")
        return vt_output

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.bert.encoder.layer[layer].attention.prune_heads(heads)


class DoubleWXDataModel(nn.Module):
    def __init__(self, args):
        super(DoubleWXDataModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        config.update({
            "fusion_layer": 6,
        })
        self.transformer = VTModel.from_pretrained(args.pretrain_path, config=config)
        self.head = nn.Linear(config.hidden_size, args.num_labels)
        # self.head = ClassificationHead(config.hidden_size, 1024, 512, args.num_labels, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.meanpooler = Model.MeanPooling()

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.65)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        # self.f1_loss = F1_Loss(num_labels=args.num_labels)

    def forward(self, input_ids, attention_mask, video_input, video_mask, labels=None):
        output = self.transformer(input_ids, attention_mask, video_input, video_mask)['last_hidden_state']
        cls_token = self.dropout(output[:,0,:])
        logits1 = self.head(self.dropout1(cls_token))
        logits2 = self.head(self.dropout2(cls_token))
        logits3 = self.head(self.dropout3(cls_token))
        logits4 = self.head(self.dropout4(cls_token))
        logits5 = self.head(self.dropout5(cls_token))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        loss = 0
        if labels is not None:
            # loss = self.loss(logits, labels)

            loss1 = self.loss(logits1, labels)
            loss2 = self.loss(logits2, labels)
            loss3 = self.loss(logits3, labels)
            loss4 = self.loss(logits4, labels)
            loss5 = self.loss(logits5, labels)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        logits = torch.softmax(logits, dim=-1)
        return logits, loss

    def loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels.squeeze(dim=-1))
        # return (loss_fn(logits, labels.squeeze(dim=-1)) + self.f1_loss(logits, labels.squeeze(dim=-1))) / 2
    
    def resize_token_embeddings(self, sz):
        self.transformer.resize_token_embeddings(sz)
