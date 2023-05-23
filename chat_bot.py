import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# モデルとトークナイザーのロード
model_name = "cyberagent/open-calm-large"
# model_name = "cyberagent/open-calm-3b"
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = AutoTokenizer.from_pretrained(model_name)

# チャットの開始
print("ボット: こんにちは! 何か質問がありますか?")

while True:
    # ユーザからの入力を受け取る
    user_input = input("あなた: ")

    # ユーザの入力をエンコードしてテンソルに変換
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    attention_mask = (input_ids != tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0).int()

    # 入力データを適切なデバイスに送る
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # モデルによる応答の生成
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=model.config.eos_token_id)

    # 生成されたテキストのデコード
    output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    print("ボット: " + output_text)
