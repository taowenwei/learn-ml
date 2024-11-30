import torch
import gpt2LLM
import tiktoken
from embedding import text2token, token2text


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
model = gpt2LLM.GPTModel(gpt2LLM.GPT_CONFIG_124M)
model.modelParamsInfo()


# Figure 4.4
def testModel():
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


# Listing 4.7 The GPT model architecture implementation
def testTextGen():
    startText = "Hello, I am"
    textTokens = text2token(tokenizer, startText)

    model.eval()
    out = gpt2LLM.generateTextSimple(
        model=model,
        textTokens=textTokens,
        maxTokenToGenerate=6,  # generate the next 6 tokens
        contextLength=gpt2LLM.GPT_CONFIG_124M["contextLength"]
    )
    endText = token2text(tokenizer, out)
    print(endText)


testModel()
testTextGen()
