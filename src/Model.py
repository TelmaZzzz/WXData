from distutils.command.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertOnlyMLMHead
import logging
import math
from Utils import F1_Loss


class ModelConfig(object):
    def __init__(self, args):
        self.pretrain_path = args.pretrain_path
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-7
        self.num_labels = args.num_labels
        self.device = args.device
        self.dropout = args.dropout


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, name="swish"):
        super(Activation, self).__init__()
        if name not in ["swish", "relu", "gelu"]:
            raise
        if name == "swish":
            self.net = Swish()
        elif name == "relu":
            self.net = nn.ReLU()
        elif name == "gelu":
            self.net = nn.GELU()
    
    def forward(self, x):
        return self.net(x)


class Dence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(Dence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class BatchDence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(BatchDence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            nn.BatchNorm1d(o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        input_mask_extended = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_extended, 1)
        sum_mask = input_mask_extended.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # [bs, 8, seqlen, 512 // 8]

    def forward(self, hidden_states, attention_mask, encoder_y=None):
        bs, len, _ = hidden_states.shape
        mixed_key_layer = self.key(hidden_states)
        if encoder_y is not None:
            mixed_query_layer = self.query(encoder_y)
            mixed_value_layer = self.value(encoder_y)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_score = attention_score + attention_mask
        attention_prob = nn.Softmax(dim=-1)(attention_score)
        attention_prob = self.drop(attention_prob)
        context_layer = torch.matmul(attention_prob, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer = context_layer.size()[:-2] + (self.hidden_size, )
        context_layer = context_layer.view(*new_context_layer)
        return context_layer


class BertVideoModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertVideoModel, self).__init__(config)
        self.config = config
        # self.video_fc = Dence(768, config.hidden_size)
        self.video_fc = nn.Linear(768, config.hidden_size)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # config.vocab_size = config.vocab_size - 3
        # self.video_embeddings = BertEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, video_input, video_mask):
        text_token_type = torch.zeros_like(attention_mask, dtype=torch.long).to(attention_mask.device)
        text_emb = self.embeddings(input_ids=input_ids, token_type_ids=text_token_type)
        
        # text input is [CLS][title][SEP][asr][SEP][ocr][SEP][SEP]
        # cls_emb = text_emb[:, 0:1, :]
        # text_emb = text_emb[:, 1:, :]
        
        # cls_mask = attention_mask[:, 0:1]
        # text_mask = attention_mask[:, 1:]

        video_feature = self.video_fc(video_input)
        # video_feature += torch.normal(0, 0.03, size=video_feature.size()).to(video_feature.device)
        video_token_type = torch.ones_like(video_mask, dtype=torch.long).to(video_mask.device)
        video_emb = self.embeddings(inputs_embeds=video_feature, token_type_ids=video_token_type)

        embedding_output = torch.cat([text_emb, video_emb], 1)
        mask = torch.cat([attention_mask, video_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        # encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask, output_hidden_states=True)
        return encoder_outputs
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


class WXDataModel(nn.Module):
    def __init__(self, args):
        super(WXDataModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        # config.update({
        #     "layer_norm_eps": args.layer_norm_eps
        # })
        self.transformer = BertVideoModel.from_pretrained(args.pretrain_path, config=config)
        self.head = nn.Linear(config.hidden_size, args.num_labels)
        # self.head = ClassificationHead(config.hidden_size, 1024, 512, args.num_labels, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.meanpooler = MeanPooling()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, video_input, video_mask, labels=None):
        output = self.transformer(input_ids, attention_mask, video_input, video_mask)['last_hidden_state']
        # cls_token = self.dropout(output[:,0,:])
        sep_mask = attention_mask[:, -1:]
        text_mask = attention_mask[:, :-1]
        mask = torch.cat([text_mask, video_mask, sep_mask], 1)
        cls_token = self.meanpooler(output, mask)
        # logits = self.head(cls_token)
        # text_mask_ = torch.zeros_like(attention_mask)
        # video_mask_ = torch.zeros_like(video_mask)
        # text_attention = torch.cat([attention_mask, video_mask_], 1)
        # text_cls = self.meanpooler(output, text_attention)
        # video_attention = torch.cat([text_mask_, video_mask], 1)
        # video_cls = self.meanpooler(output, video_attention)
        # cls_token = torch.cat([text_cls, video_cls], 1)


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
    
    def resize_token_embeddings(self, sz):
        self.transformer.resize_token_embeddings(sz)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_size, in_size_0, in_size_1, output_size, dropout):
        super().__init__()
        # self.norm= nn.BatchNorm1d(input_size)
        self.dense = Dence(input_size, in_size_0)
        # self.dense = BatchDence(input_size, in_size_0)
        self.dropout = nn.Dropout(dropout)
        self.dense_1 = Dence(in_size_0, in_size_1)
        # self.dense_1 = BatchDence(in_size_0, in_size_1)
        # self.dropout_1 = nn.Dropout(dropout)  
        # self.out_proj = nn.Linear(


class WXDataPretrainModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        self.bert = BertVideoModel(config)
        self.cls = BertOnlyMLMHead(config)
        # ADD
        self.meanpooler = MeanPooling()
        self.text_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.video_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.vtm_head = nn.Linear(config.hidden_size, 2)
        # /ADD
    
    def forward(self, input_ids, attention_mask, video_input, video_mask, input_ids_r, labels):
        # MLM
        bert_output = self.bert(input_ids, attention_mask, video_input, video_mask)['last_hidden_state']
        mlm_output = self.cls(bert_output[:,:input_ids.size(1),:])
        mlm_loss = self.mlm_loss(mlm_output, labels)
        # VTC
        # with torch.no_grad():
        #     self.temp.clamp_(0.001, 0.5)
        vtc_bert_output = self.bert(input_ids_r.clone(), attention_mask, video_input.clone(), video_mask)['last_hidden_state']
        text_mask = attention_mask
        text_mask_ = torch.zeros_like(text_mask)
        video_mask_ = torch.zeros_like(video_mask)
        text_attention = torch.cat([text_mask, video_mask_], dim=1)
        video_attention = torch.cat([text_mask_, video_mask], dim=1)
        text_feat = F.normalize(self.text_proj(self.meanpooler(vtc_bert_output, text_attention)), dim=-1)
        video_feat = F.normalize(self.video_proj(self.meanpooler(vtc_bert_output, video_attention)), dim=-1)
        sim_v2t = video_feat @ text_feat.t() / self.temp
        sim_t2v = text_feat @ video_feat.t() / self.temp
        sim_target = torch.arange(input_ids.size(0)).to(input_ids.device)
        vtc_loss = self.vtc_loss(sim_v2t, sim_t2v, sim_target)
        # VTM
        bs = input_ids.size(0)
        vtm_input_ids = input_ids_r.clone()
        vtm_video_input = video_input.clone()
        pos_bert_output = vtc_bert_output
        # pos_bert_output = self.bert(vtm_input_ids, attention_mask, vtm_video_input, video_mask)
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
        video_neg_bert_output = self.bert(vtm_input_ids, attention_mask, video_neg, video_att_neg)['last_hidden_state']
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


class WXDataModelV2(nn.Module):
    def __init__(self, args):
        super(WXDataModelV2, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        # config.update({
        #     "layer_norm_eps": args.layer_norm_eps
        # })
        self.transformer = BertVideoModel.from_pretrained(args.pretrain_path, config=config)
        self.head = nn.Linear(config.hidden_size * 2, args.num_labels)
        # self.head = ClassificationHead(config.hidden_size, 1024, 512, args.num_labels, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.meanpooler = MeanPooling()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        # self.f1_loss = F1_Loss(num_labels=args.num_labels)

    def forward(self, input_ids, attention_mask, video_input, video_mask, labels=None):
        output = self.transformer(input_ids, attention_mask, video_input, video_mask)['last_hidden_state']
        cls_info = self.dropout(output[:,0,:])
        mask = torch.cat([attention_mask, video_mask], 1)
        cls_token = self.meanpooler(output, mask)
        cls_token = torch.cat([cls_token, cls_info], dim=-1)

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


class WXDataModelV3(nn.Module):
    def __init__(self, args):
        super(WXDataModelV3, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        # config.update({
        #     "layer_norm_eps": args.layer_norm_eps
        # })
        self.transformer = BertVideoModel.from_pretrained(args.pretrain_path, config=config)
        self.head = nn.Linear(config.hidden_size * 5, args.num_labels)
        # self.head = ClassificationHead(config.hidden_size, 1024, 512, args.num_labels, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.meanpooler = MeanPooling()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        # self.f1_loss = F1_Loss(num_labels=args.num_labels)

    def forward(self, input_ids, attention_mask, video_input, video_mask, labels=None):
        output = self.transformer(input_ids, attention_mask, video_input, video_mask)['hidden_states']
        l1, l2, l3, l4 = output[-4], output[-3], output[-2], output[-1]
        cls_info = torch.cat([l1[:,0,:], l2[:,0,:], l3[:,0,:], l4[:,0,:]], dim=-1)
        # cls_info = self.dropout(output[:,0,:])
        cls_info = self.dropout(cls_info)
        # sep_mask = attention_mask[:, -1:]
        # text_mask = attention_mask[:, :-1]
        mask = torch.cat([attention_mask, video_mask], 1)
        # cls_token = self.meanpooler(output, mask)
        cls_token = self.meanpooler(l4, mask)
        cls_token = torch.cat([cls_token, cls_info], dim=-1)
        # logits = self.head(cls_token)
        # text_mask_ = torch.zeros_like(attention_mask)
        # video_mask_ = torch.zeros_like(video_mask)
        # text_attention = torch.cat([attention_mask, video_mask_], 1)
        # text_cls = self.meanpooler(output, text_attention)
        # video_attention = torch.cat([text_mask_, video_mask], 1)
        # video_cls = self.meanpooler(output, video_attention)
        # cls_token = torch.cat([text_cls, video_cls], 1)


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
