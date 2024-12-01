import torch
import gpt2LLM
from embedding import TokenDataLoader, text2token, token2text
import tiktoken

torch.manual_seed(123)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("gpt2_model.pth", map_location=device)
tokenizer = tiktoken.get_encoding("gpt2")

gptConfig = {**gpt2LLM.GPT_CONFIG_124M}
gptConfig["qkvBias"] = True
gpt2 = gpt2LLM.GPTModel(gptConfig)
gpt2.to(device)
gpt2.load_state_dict(torch.load("gpt2_model.pth", map_location=device))

gpt2.eval()
output = gpt2LLM.generateTextSimple(
    model=gpt2,
    textTokens=text2token(tokenizer, "Every effort moves you").to(device),
    maxTokenToGenerate=25,
    contextLength=gptConfig["contextLength"],
)
print("Output text:\n", token2text(tokenizer, output))
