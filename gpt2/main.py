import torch
import gpt2LLM
import tiktoken


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
model = gpt2LLM.GPTModel(gpt2LLM.GPT_CONFIG_124M)
model.modelParamsInfo()


# Listing 4.8 A function for the GPT model to generate text
def generate_text_simple(model, idx,
                         maxTokenLen, contextLength):
    for _ in range(maxTokenLen):
        idxChunk = idx[:, -contextLength:]
        with torch.no_grad():
            logits = model(idxChunk)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idxNext = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idxNext), dim=1)
    return idx


# based on the startText, generate the next two words
startText = "Hello, I am"
encoded = tokenizer.encode(startText)
print("encoded:", encoded)
idx = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", idx.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=idx,
    maxTokenLen=4 + 2,
    contextLength=gpt2LLM.GPT_CONFIG_124M["contextLength"]
)
print("Output:", out)
print("Output length:", len(out[0]))
endText = tokenizer.decode(out.squeeze(0).tolist())
print(endText)
