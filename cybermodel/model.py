import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_model(model_name="cyberagent/open-calm-large"):
    """
    プリトレーニング済みのモデルとトークナイザをロードします。

    Args:
        model_name (str): プリトレーニング済みのモデルの名前。デフォルトは "cyberagent/open-calm-large"。

    Returns:
        model: ロードしたモデル。GPUが利用可能な場合はGPUに、そうでない場合はCPUに配置されます。
        tokenizer: ロードしたトークナイザ。
    """
    # プリトレーニング済みのモデルをロードし、デバイス（GPUまたはCPU）に配置します。
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # プリトレーニング済みのモデルに対応するトークナイザをロードします。
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # モデルとトークナイザを返します。
    return model, tokenizer
