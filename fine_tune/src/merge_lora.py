import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = r"D:\python_code\pretrained\Qwen\Qwen2.5-3B-Instruct"
lora_path = r"D:\python_code\model\lora_model"
merge_path = r"D:\python_code\model\merged_model"

# 1️⃣ 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"   # merge 建议用 cpu
)

# 2️⃣ 加载 LoRA
model = PeftModel.from_pretrained(model, lora_path)

# 3️⃣ 合并
model = model.merge_and_unload()

# 4️⃣ 保存完整模型
model.save_pretrained(merge_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merge_path)

print("✅ 合并完成")