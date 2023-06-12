import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2ForSequenceClassification
import math
from config import Config


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_factor):
        super(ProjectionBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, expansion_factor * in_dim)
        self.linear2 = nn.Linear(expansion_factor * in_dim, expansion_factor * out_dim)
        self.linear3 = nn.Linear(expansion_factor * out_dim, out_dim)

    def forward(self, x):
        y = self.linear1(x)
        y = self.linear2(y)
        y = self.linear3(y)
        return y


class UtEncoder(nn.Module):
    def __init__(self, config: Config):
        super(UtEncoder, self).__init__()
        self.modal = config.modal
        self.device = config.device
        roberta_model_path = "princeton-nlp/sup-simcse-roberta-large"
        wav2vec_model_path = "Zahra99/wav2vec2-base-finetuned-iemocap6"
        if self.modal != "audio":
            self.roberta = AutoModel.from_pretrained(roberta_model_path, local_files_only=False)
        if self.modal != "text":
            self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_model_path, local_files_only=False)
        if self.modal == "bimodal":
            self.text_embedding_dim = 1024
            self.audio_embedding_dim = 768
            self.embedding_dim = config.uttr_embedding_dim
            self.embedding_length = 256 + 299
            self.text_projection_block = ProjectionBlock(self.text_embedding_dim, self.embedding_dim, config.projection_expansion_factor)
            self.audio_projection_block = ProjectionBlock(self.audio_embedding_dim, self.embedding_dim, config.projection_expansion_factor)
            self.position_embedding = nn.Embedding(self.embedding_length, self.embedding_dim)
            self.uttr_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=config.uttr_encoder_nhead, batch_first=True)
            self.uttr_encoder = nn.TransformerEncoder(self.uttr_encoder_layer, config.uttr_encoder_layers)

    def forward(self, encode_texts, encode_audios):
        if self.modal != "audio":
            encode_texts = torch.squeeze(encode_texts, 0)
            text_embeds = self.roberta(input_ids=encode_texts).last_hidden_state
            target_token_index = text_embeds.size()[1] - 2
            if self.modal == "text":
                encode_uttrs = text_embeds[:, -2, :]
                encode_uttrs = torch.unsqueeze(encode_uttrs, 0)
                return encode_uttrs
        encode_audios = torch.squeeze(encode_audios, 0)
        audio_embeds = self.wav2vec(encode_audios, output_hidden_states=True).hidden_states[-1]
        if self.modal == "audio":
            encode_uttrs = torch.mean(audio_embeds, -2)
            encode_uttrs = torch.unsqueeze(encode_uttrs, 0)
            return encode_uttrs
        text_embeds = self.text_projection_block(text_embeds)
        audio_embeds = self.audio_projection_block(audio_embeds)
        uttr_embeds = torch.hstack([text_embeds, audio_embeds])
        position_ids = torch.arange(uttr_embeds.size()[1], dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand([uttr_embeds.size()[0], uttr_embeds.size()[1]])
        position_embeds = self.position_embedding(position_ids)
        uttr_embeds = uttr_embeds + position_embeds
        uttr_embeds = self.uttr_encoder(uttr_embeds)
        encode_uttrs = uttr_embeds[:, target_token_index, :]
        encode_uttrs = torch.unsqueeze(encode_uttrs, 0)
        return encode_uttrs


class LaRNet(nn.Module):
    def __init__(self, config: Config):
        super(LaRNet, self).__init__()
        self.device = config.device
        self.use_combine_loss = config.use_combine_loss
        if config.modal == "text":
            self.embedding_dim = 1024
        elif config.modal == "audio":
            self.embedding_dim = 768
        else:
            self.embedding_dim = config.uttr_embedding_dim
        self.ut_encoder = UtEncoder(config)
        self.max_uttr_num = config.max_uttrs_num
        if config.dataset == "meld":
            self.max_uttr_num = min(self.max_uttr_num, 40)
        self.position_embedding = nn.Embedding(self.max_uttr_num, self.embedding_dim)
        self.use_lar_attention = config.use_lar_attention
        self.pre_alpha = config.lar_attention_pre_alpha
        self.post_alpha = config.lar_attention_post_alpha
        self.uttrs_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=config.uttrs_encoder_nhead, batch_first=True)
        self.uttrs_encoder = nn.TransformerEncoder(self.uttrs_encoder_layer, num_layers=config.uttrs_encoder_layers)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.embedding_dim, config.num_classes)

    def forward(self, encode_texts, encode_audios):
        encode_uttrs = self.ut_encoder(encode_texts, encode_audios)
        position_ids = torch.arange(encode_uttrs.size()[1], dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand([encode_uttrs.size()[0], encode_uttrs.size()[1]])
        position_embeds = self.position_embedding(position_ids)
        uttr_embeds = encode_uttrs + position_embeds
        mask = generate_lar_attention_mask(uttr_embeds.size()[1], self.pre_alpha, self.post_alpha)
        mask = mask.to(self.device)
        if self.use_lar_attention:
            uttr_embeds = self.uttrs_encoder(uttr_embeds, mask=mask)
        else:
            uttr_embeds = self.uttrs_encoder(uttr_embeds)
        dialog_embeds = encode_uttrs + uttr_embeds
        y = self.linear(dialog_embeds)
        y = self.relu(y)
        y = self.classifier(y)
        if not self.use_combine_loss:
            return y
        else:
            return encode_uttrs, y


def generate_lar_attention_mask(seq_len, pre_alpha, post_alpha):
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    for i in range(seq_len):
        j = 0
        while j <= i:
            mask[i][j] = lar_attention_weight(i-j, alpha=pre_alpha)
            j += 1
        while j < seq_len:
            mask[i][j] = lar_attention_weight(j-i, alpha=post_alpha)
            j += 1
    return mask


def lar_attention_weight(distance, alpha):
    return (-math.exp(distance) + 1) * alpha
