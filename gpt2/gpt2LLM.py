import torch
import torch.nn as nn
from attentions import MultiHeadAttention2


GPT_CONFIG_124M = {
    "vocabSize": 50257,         # Vocabulary size
    "contextLength": 1024,      # Context length
    "embeddingDims": 768,       # Embedding dimension
    "numHeads": 12,             # Number of attention heads
    "numLayers": 12,            # Number of layers
    "dropRate": 0.1,            # Dropout rate
    "qkvBias": False            # Query-Key-Value bias
}


# Listing 4.2 A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, embeddingDims):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embeddingDims))
        self.shift = nn.Parameter(torch.zeros(embeddingDims))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# Listing 4.7 The GPT model architecture implementation
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tokenEmbeddings = nn.Embedding(
            cfg["vocabSize"], cfg["embeddingDims"])
        self.positionEmbeddings = nn.Embedding(
            cfg["contextLength"], cfg["embeddingDims"])
        self.dropoutEmbeddings = nn.Dropout(cfg["dropRate"])
        self.transformerBlocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["numLayers"])])
        self.finalNorm = LayerNorm(cfg["embeddingDims"])
        self.outHead = nn.Linear(
            cfg["embeddingDims"], cfg["vocabSize"], bias=False
        )

    def forward(self, inputsText):
        batchSize, seqLen = inputsText.shape
        tokenEmbeddingseds = self.tokenEmbeddings(inputsText)
        positionEmbeddingseds = self.positionEmbeddings(
            torch.arange(seqLen, device=inputsText.device)
        )
        x = tokenEmbeddingseds + positionEmbeddingseds
        x = self.dropoutEmbeddings(x)
        x = self.transformerBlocks(x)
        x = self.finalNorm(x)
        logits = self.outHead(x)
        return logits

    def modelParamsInfo(self):
        print("Model details:")
        totalParams = sum(p.numel() for p in self.parameters())
        print(f"  Total number of parameters: {totalParams:,}")
        print("  Token embedding layer shape:",
              self.tokenEmbeddings.weight.shape)
        print("  Output layer shape:", self.outHead.weight.shape)
        gpt2TotalParams = (
            totalParams - sum(p.numel()
                              for p in self.outHead.parameters())
        )
        print(
            f"  Number of trainable parameters considering weight tying: {gpt2TotalParams:,}")
        totalSizeBytes = totalParams * 4
        print(
            f"  Total size of the model: {totalSizeBytes / (1024 * 1024):.2f} MB")


# Listing 4.6 The transformer block component of GPT
class TransformerBlock(nn.Module):

    # Listing 4.3 An implementation of the GELU activation function
    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))

    # Listing 4.4 A feed forward neural network module
    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["embeddingDims"], 4 * cfg["embeddingDims"]),
                TransformerBlock.GELU(),
                nn.Linear(4 * cfg["embeddingDims"], cfg["embeddingDims"]),
            )

        def forward(self, x):
            return self.layers(x)

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention2(
            inputDims=cfg["embeddingDims"],
            outputDims=cfg["embeddingDims"],
            contextLength=cfg["contextLength"],
            numHeads=cfg["numHeads"],
            dropout=cfg["dropRate"],
            qkvBias=cfg["qkvBias"])
        self.ff = TransformerBlock.FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embeddingDims"])
        self.norm2 = LayerNorm(cfg["embeddingDims"])
        self.dropShortcut = nn.Dropout(cfg["dropRate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropShortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropShortcut(x)
        x = x + shortcut
        return x


def generateTextSimple(model, textTokens,
                       maxTokenToGenerate, contextLength):
    for _ in range(maxTokenToGenerate):
        chunk = textTokens[:, -contextLength:]
        with torch.no_grad():
            logits = model(chunk)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        token = torch.argmax(probas, dim=-1, keepdim=True)
        textTokens = torch.cat((textTokens, token), dim=1)
    return textTokens
