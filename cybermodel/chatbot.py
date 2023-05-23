def run_bot(model, tokenizer, first_text, you_text, max_count=10):
    # チャットの開始
    print(first_text) # "ボット: こんにちは! 何か質問がありますか?"

    count = 0
    while True:
        if max_count <= count:
            break
        
        # ユーザからの入力を受け取る
        user_input = input(you_text)  # "あなた: "

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
        
        count += 1
