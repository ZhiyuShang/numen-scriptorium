import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from peft import PeftModel
import time
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = os.path.join("outputs", "qwen2_5_1_5b_boh_qlora", "checkpoint-1400")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, instruction: str, user_input: str = "", max_new_tokens: int = 256):
    if user_input:
        prompt = f"指令：{instruction}\n输入：{user_input}\n回答："
    else:
        prompt = f"指令：{instruction}\n回答："

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "回答：" in text:
        text = text.split("回答：", 1)[1]
    return text.strip()


if __name__ == "__main__":
    tokenizer, model = load_model()
    start_time = time.time()

    # Example: English -> Chinese translation
    instr = "你是一名翻译者，需要将英文文本翻译为中文。\n"
    "要求：\n"
    "1. 保留所有专有名词（包括人物、地名），不要将它们翻译为《司辰之书》；\n"
    "2. 在保证忠实的前提下，让整体语气更接近游戏《司辰之书》，适度使用象征与暗示。\n"
    "请翻译下面这段英文故事："
    
    #'将下面的英文文本翻译为中文，保持原作的神秘主义与哲思感，并保留隐喻与象征的味道。'
    #"将下面的英文故事段落翻译为中文，保持《司辰之书》式的叙事风格，善用象征与暗示。"
    user = (
       "Name: The Sun's Design\nType: tablet\n"
       "Description: A scorched slab of black corundum, minutely scratched on every side with intricate ideoglyphs.\n "
       "In the city of Emesa, beneath the Church of the Holy Belt, in a sarcophagus of black corundum, Elagabalus lies: accursed of Janus, neither Long nor mortal, neither man nor woman, neither a liar nor a speaker of truth, neither real nor imagined. On his light-suffused skin is made manifest the Sun-in-Splendour's grand design...\n Elagabalus is the source of one-half of this text. The source of the other-half is obscure, but its power is evident. It is impossible to be certain if the Sun really planned for us to enter Eternity. It is impossible to be sure if the Grail, the Vagabond, and the Forge, stole this birthright from us, or saved us from it. But there is a great secret here."
    )

    out = generate(tokenizer, model, instr, user, max_new_tokens=500)
    print("=== English -> Chinese Translation ===")
    print(out)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")