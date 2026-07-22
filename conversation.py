from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "outputs/flan-t5-large_yahoo10000_ep3_bs1_acc8/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def ask_model(question):
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=4,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(ask_model("Why is the sky blue?"))