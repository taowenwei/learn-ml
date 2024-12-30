import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, textInput, tokenizer, maxLength, stride):
        self.inputIds = []
        self.targetIds = []
        tokens = tokenizer.encode(textInput)

        # 2.6 Data sampling with a sliding window
        for i in range(0, len(tokens) - maxLength, stride):
            inputChunk = tokens[i:i + maxLength]
            targetChunk = tokens[i + 1: i + maxLength + 1]
            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))

    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, idx):
        return self.inputIds[idx], self.targetIds[idx]


class TokenDataLoader(DataLoader):
    def __init__(self, textData, batchSize=4, maxLength=256,
                 stride=128, shuffle=True, dropLast=True,
                 numWorkers=0):
        tokenizer = tiktoken.get_encoding("gpt2")  # 1
        dataset = GPTDatasetV1(textData, tokenizer, maxLength, stride)
        super().__init__(
            dataset=dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            drop_last=dropLast,
            num_workers=numWorkers)


class TextEmbedding:
    def __init__(self, textData, batchSize=4, maxLength=256,
                 stride=128, shuffle=True, dropLast=True,
                 numWorkers=0):
        self.dataLoader = TokenDataLoader(textData,
                                batchSize=batchSize,
                                maxLength=maxLength,
                                stride=stride,
                                shuffle=shuffle,
                                dropLast=dropLast,
                                numWorkers=numWorkers)

        # BPE tokenizer (tiktoken.get_encoding("gpt2")) vocabulary size is 50257
        vocabSize = 50257
        outputDim = 256
        self.tokenEmbeddingLayer = torch.nn.Embedding(vocabSize, outputDim)
        self.positionEmbeddingLayer = torch.nn.Embedding(maxLength, outputDim)
        self.positionEmbeddings = self.positionEmbeddingLayer(
            torch.arange(maxLength))

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
        print("Position embeddings:\n", self.positionEmbeddings)
        # print("\nPosition embeddings shape:\n", self.positionEmbeddings.shape)

        # 2.8 Encoding word positions
        inputEmbeddings = tokenEmbeddings + self.positionEmbeddings
        print("Input embeddings:\n", inputEmbeddings)


def text2token(tokenizer, text):
    encoded = tokenizer.encode(text)
    encodedTensor = torch.tensor(encoded).unsqueeze(0)
    return encodedTensor


def token2text(tokenizer, token):
    flat = token.squeeze(0)
    return tokenizer.decode(flat.tolist())
