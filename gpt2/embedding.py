import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenIds = []
        with open("theVerdict.txt", "r", encoding="utf-8") as f:
            rawText = f.read()
            # 2.5 Byte pair encoding
            self.tokenIds = self.tokenizer.encode(rawText)

    def tokens(self):
        return self.tokenIds


class GPTDatasetV1(Dataset):
    def __init__(self, tokenizer, maxLength, stride):
        self.inputIds = []
        self.targetIds = []
        tokenIds = tokenizer.tokens()

        # 2.6 Data sampling with a sliding window
        for i in range(0, len(tokenIds) - maxLength, stride):
            inputChunk = tokenIds[i:i + maxLength]
            targetChunk = tokenIds[i + 1: i + maxLength + 1]
            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))

    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, idx):
        return self.inputIds[idx], self.targetIds[idx]


class TextEmbedding:
    def __init__(self, batchSize=4, maxLength=256,
                 stride=128, shuffle=True, dropLast=True,
                 numWorkers=0):
        self.tokenizer = Tokenizer()
        self.dataLoader = DataLoader(
            dataset=GPTDatasetV1(self.tokenizer, maxLength, stride),
            batch_size=batchSize,
            shuffle=shuffle,
            drop_last=dropLast,
            num_workers=numWorkers)

        torch.manual_seed(123)
        # BPE tokenizer (tiktoken.get_encoding("gpt2")) vocabulary size is 50257
        vocabSize = 50257
        outputDim = 256
        self.tokenEmbeddingLayer = torch.nn.Embedding(vocabSize, outputDim)
        self.positionEmbeddingLayer = torch.nn.Embedding(maxLength, outputDim)
        self.positionEmbeddings = self.positionEmbeddingLayer(
            torch.arange(maxLength))
        print("Position embeddings:\n", self.positionEmbeddings)
        # print("\nPosition embeddings shape:\n", self.positionEmbeddings.shape)

    def dataIter(self):
        return iter(self.dataLoader)

    def processOneBatch(self, dataIter):
        inputs, targets = next(dataIter)
        print("Token IDs:\n", inputs)
        # print("\nInputs shape:\n", inputs.shape)
        
        # 2.7 Creating token embeddings
        tokenEmbeddings = self.tokenEmbeddingLayer(inputs)
        print("Embeddings:\n", tokenEmbeddings)
        # print("\nEmbeddings shape:\n", tokenEmbeddings.shape)
        
        # 2.8 Encoding word positions
        inputEmbeddings = tokenEmbeddings + self.positionEmbeddings
        print("Input embeddings:\n", inputEmbeddings)


embedding = TextEmbedding(
    batchSize=1, maxLength=4, stride=1, shuffle=False)
dataIter = embedding.dataIter()
embedding.processOneBatch(dataIter)
