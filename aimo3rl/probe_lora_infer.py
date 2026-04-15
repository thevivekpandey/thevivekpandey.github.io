import torch
from peft import PeftModel
from transformers import AutoTokenizer, Mxfp4Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
ADAPTER_PATH = "/home/ubuntu/qlora/lora_adapters_final"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [
        {"role": "developer", "content": "You are a helpful math assistant. Answer briefly."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    model = GptOssForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=Mxfp4Config(pre_quantized=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    prompt_ids = prompt_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            prompt_ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002, 200012],
        )

    completion_ids = output[0, prompt_ids.shape[1]:]
    print(tokenizer.decode(completion_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
