import os
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPreTrainedModel, BertOnlyMLMHead
import xbert
import torch
import torch.nn as nn
import torch.nn.functional as F
import Model
import copy


class VTModel(BertPreTrainedModel):
    def __init__(self, config):
        super(VTModel, self).__init__(config)
        # config = AutoConfig.from_pretrained(args.pretrain_path)
        self.config = config
        self.bert = xbert.BertModel(config)
        # self.bert = xbert.BertModel.from_pretrained(config.pretrain_path, config=config)
        self.vl_config = copy.deepcopy(self.config)
        self.vl_config.update({
            "num_hidden_layers": 4,
        })
        self.vl_encoder = BertEncoder(self.vl_config)
        # self.vl_encoder.apply(self._init_weights)
        self.init_weights()
        # self.video_embeddings = BertEmbeddings()
    
    def forward(self, input_ids, attention_mask, video_input, video_mask):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, mode="text")
        text_embeds = text_output.last_hidden_state
        mask = video_mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        video_embeds = self.vl_encoder(video_input, attention_mask=mask, output_hidden_states=True)["last_hidden_state"]
        vt_output = self.bert(encoder_embeds=text_embeds, attention_mask=attention_mask, \
            encoder_hidden_states=video_embeds, encoder_attention_mask=video_mask, return_dict=True, mode="fusion")
        return vt_output, video_embeds

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
            "encoder_width": config.hidden_size,
            # "pretrain_path": args.pretrain_path,
        })
        self.transformer = VTModel.from_pretrained(args.pretrain_path, config=config)
        # self.transformer = VTModel(config)
        self.head = nn.Linear(config.hidden_size * 2, args.num_labels)
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
        output, video_embeds = self.transformer(input_ids, attention_mask, video_input, video_mask)
        output = output["last_hidden_state"]
        cls_token = self.dropout(output[:,0,:])
        video_token = self.meanpooler(video_embeds, video_mask)
        cls_token = torch.cat([cls_token, video_token], dim=-1)

        logits1 = self.head(self.dropout1(cls_token))
        logits2 = self.head(self.dropout2(cls_token))
        logits3 = self.head(self.dropout3(cls_token))
        logits4 = self.head(self.dropout4(cls_token))
        logits5 = self.head(self.dropout5(cls_token))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        # logits = self.head(cls_token)
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


class VTPretrainModel(BertPreTrainedModel):
    def __init__(self, config):
        super(VTPretrainModel, self).__init__(config)
        self.config = config
        self.bert = xbert.BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.vl_config = copy.deepcopy(self.config)
        self.vl_config.update({
            "num_hidden_layers": 4,
        })
        self.vl_encoder = BertEncoder(self.vl_config)
        self.meanpooler = Model.MeanPooling()
        self.text_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.video_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.vtm_head = nn.Linear(config.hidden_size, 2)
        # self.vl_encoder.apply(self._init_weights)
        # self.init_weights()

    def _forward(self, input_ids, attention_mask, video_input, video_mask):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, mode="text")
        text_embeds = text_output.last_hidden_state
        mask = video_mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        video_embeds = self.vl_encoder(video_input, attention_mask=mask, output_hidden_states=True)["last_hidden_state"]
        vt_output = self.bert(encoder_embeds=text_embeds, attention_mask=attention_mask, \
            encoder_hidden_states=video_embeds, encoder_attention_mask=video_mask, return_dict=True, mode="fusion")
        return vt_output

    def forward(self, input_ids, attention_mask, video_input, video_mask, input_ids_r, labels):
        # MLM
        bert_output = self._forward(input_ids, attention_mask, video_input, video_mask)["last_hidden_state"]
        mlm_output = self.cls(bert_output[:,:input_ids.size(1),:])
        mlm_loss = self.mlm_loss(mlm_output, labels)
        # VTC
        # with torch.no_grad():
        #     self.temp.clamp_(0.001, 0.5)
        text_output = self.bert(input_ids=input_ids_r, attention_mask=attention_mask, return_dict=True, mode="text")
        text_embeds = text_output.last_hidden_state
        mask = video_mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        video_embeds = self.vl_encoder(video_input, attention_mask=mask, output_hidden_states=True)["last_hidden_state"]
        text_feat = F.normalize(self.text_proj(self.meanpooler(text_embeds, attention_mask)), dim=-1)
        video_feat = F.normalize(self.video_proj(self.meanpooler(video_embeds, video_mask)), dim=-1)
        sim_v2t = video_feat @ text_feat.t() / self.temp
        sim_t2v = text_feat @ video_feat.t() / self.temp
        sim_target = torch.arange(input_ids.size(0)).to(input_ids.device)
        vtc_loss = self.vtc_loss(sim_v2t, sim_t2v, sim_target)
        # VTM
        bs = input_ids.size(0)
        vtm_input_ids = input_ids_r.clone()
        vtm_video_input = video_input.clone()
        # pos_bert_output = vtc_bert_output
        pos_bert_output = self.bert(encoder_embeds=text_embeds, attention_mask=attention_mask, \
            encoder_hidden_states=video_embeds, encoder_attention_mask=video_mask, return_dict=True, mode="fusion")['last_hidden_state']
        # with torch.no_grad():
        #     weight_t2v = torch.ones((bs,bs), dtype=torch.float, device=input_ids.device)
        #     # weight_t2v = F.softmax(sim_t2v, dim=1) + 1e-4
        #     weight_t2v.fill_diagonal_(0)
        #     weight_v2t = torch.ones((bs,bs), dtype=torch.float, device=input_ids.device)
        #     # weight_v2t = F.softmax(sim_v2t, dim=1) + 1e-4
        #     weight_v2t.fill_diagonal_(0)
        video_neg = []
        video_att_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weight_t2v[b], 1).item()
        #     video_neg.append(vtm_video_input[neg_idx])
        #     video_att_neg.append(video_mask[neg_idx])
        for b in range(bs - 1, -1, -1):
            video_neg.append(vtm_video_input[b])
            video_att_neg.append(video_mask[b])
        video_neg = torch.stack(video_neg, dim=0)
        video_att_neg = torch.stack(video_att_neg, dim=0)
        video_neg_bert_output = self._forward(vtm_input_ids, attention_mask, video_neg, video_att_neg)['last_hidden_state']
        # text_neg = []
        # text_att_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weight_v2t[b], 1).item()
        #     text_neg.append(vtm_input_ids[neg_idx])
        #     text_att_neg.append(attention_mask[neg_idx])
        # text_neg = torch.stack(text_neg, dim=0)
        # text_att_neg = torch.stack(text_att_neg, dim=0)
        # text_neg_bert_output = self.bert(text_neg, text_att_neg, vtm_video_input, video_mask)
        # vtm_output = torch.cat([pos_bert_output[:,0,:], video_neg_bert_output[:,0,:], text_neg_bert_output[:,0,:]], dim=0)
        vtm_output = torch.cat([pos_bert_output[:,0,:], video_neg_bert_output[:,0,:]], dim=0)
        vtm_output = self.vtm_head(vtm_output)
        vtm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(bs, dtype=torch.long)], dim=0).to(input_ids.device)
        vtm_loss = self.vtm_loss(vtm_output, vtm_labels)
        # logging.info(f"mlm_loss: {mlm_loss} vtc_loss: {vtc_loss} vtm_loss: {vtm_loss}")
        return mlm_output, (mlm_loss + vtc_loss + vtm_loss) / 3
        # return mlm_output, (mlm_loss + vtm_loss) / 2
    
    def mlm_loss(self, mlm_logits, mlm_labels):
        loss_fn = nn.CrossEntropyLoss()
        _, _, vocab_size = mlm_logits.shape
        return loss_fn(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))
    
    def vtc_loss(self, sim_v2t, sim_t2v, sim_labels):
        loss_fn = nn.CrossEntropyLoss()
        return (loss_fn(sim_v2t, sim_labels) + loss_fn(sim_t2v, sim_labels)) / 2
    
    def vtm_loss(self, vtm_logits, vtm_labels):
        return F.cross_entropy(vtm_logits, vtm_labels)
