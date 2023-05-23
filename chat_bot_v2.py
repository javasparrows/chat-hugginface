# from cybermodel.model import set_model
# from cybermodel.chatbot import run_bot
from cybermodel import set_model, run_bot


if __name__ == "__main__":
    model_name = "cyberagent/open-calm-large"
    model, tokenizer = set_model(model_name=model_name)

    first_text = "ボット: ハロー！なんだい？"
    you_text = "あなた："
    run_bot(model, tokenizer, first_text, you_text, max_count=1)