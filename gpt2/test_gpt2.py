import torch
import gpt2LLM
import tiktoken


torch.manual_seed(123)

model = gpt2LLM.GPTModel(gpt2LLM.GPT_CONFIG_124M)
model.modelParamsInfo()

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print("\nInput batch:\n", batch)

out = model(batch)
print("\nOutput shape:", out.shape)
print(out)
