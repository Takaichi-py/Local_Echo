# model.py
import torch
import torch.nn as nn
import math
import spacy
nlp = spacy.load("ja_core_news_md")
from torchtext.vocab import Vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
def ja_tokenizer(text):
    return [tok.text for tok in nlp.tokenizer(text)]
# Transformer の構成 (学習時と同じ)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.d_model = d_model
    def forward(self, x):
        seq_len = x.size(0)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.nhead = nhead
    def forward(self, src, tgt):
        src_embed = self.pos_enc(self.src_emb(src))
        memory = self.transformer_encoder(src_embed)
        tgt_embed = self.pos_enc(self.tgt_emb(tgt))
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output
def load_trained_model(model_path, src_vocab_size, tgt_vocab_size):
    """
    学習済みの state_dict を TransformerModel に読み込み
    """
    model = TransformerModel(src_vocab_size, tgt_vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
def standard_to_kansai_inference(model, src_vocab, tgt_vocab, text, max_len=50):
    """
    1文のみのGreedyデコード推論
    """
    model.eval()
    tokens = ja_tokenizer(text)
    src_tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]
    src_ids = [src_vocab[tok] for tok in src_tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)
    # Encoder
    with torch.no_grad():
        memory = model.transformer_encoder(
            model.pos_enc(model.src_emb(src_tensor))
        )
    # Decoder (Greedy)
    generated_tokens = [BOS_TOKEN]
    for _ in range(max_len):
        tgt_ids = [tgt_vocab[tok] for tok in generated_tokens]
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(1).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0)).to(device)
        with torch.no_grad():
            dec_out = model.transformer_decoder(
                model.pos_enc(model.tgt_emb(tgt_tensor)),
                memory,
                tgt_mask=tgt_mask
            )
            logits = model.fc_out(dec_out)
            next_token_id = logits[-1, 0, :].argmax(dim=-1).item()
        next_word = tgt_vocab.lookup_token(next_token_id)
        generated_tokens.append(next_word)
        if next_word == EOS_TOKEN:
            break
    # <sos>, <eos> を除去
    if generated_tokens[-1] == EOS_TOKEN:
        generated_tokens = generated_tokens[1:-1]
    else:
        generated_tokens = generated_tokens[1:]
    return "".join(generated_tokens)