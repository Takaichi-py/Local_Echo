# app.py
from flask import Flask, request, render_template
import torch
from model import (
    device,
    load_trained_model,
    standard_to_kansai_inference
)
app = Flask(__name__)
# 学習済みvocabをロード (train.py で保存したファイル)
src_vocab = torch.load("src_vocab.pth")
tgt_vocab = torch.load("tgt_vocab.pth")
SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
# 学習済みモデルをロード
model = load_trained_model(
    "standard_to_kansai_transformer.pt",  # train.pyで保存したモデルファイル
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE
)
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    同じページ (index.html) 上で標準語入力 & 関西弁を表示する例
    """
    kansai_text = ""
    standard_text = ""
    if request.method == 'POST':
        standard_text = request.form.get("text", "")
        kansai_text = standard_to_kansai_inference(model, src_vocab, tgt_vocab, standard_text)
    return render_template("index.html",
                           input_text=standard_text,
                           result=kansai_text)
if __name__ == '__main__':
    app.run(debug=True)