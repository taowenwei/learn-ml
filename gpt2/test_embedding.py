import torch
from embedding import TextEmbedding

torch.manual_seed(123)

with open("theVerdict.txt", "r", encoding="utf-8") as f:
    rawText = f.read()

    embedding = TextEmbedding(rawText,
                              batchSize=1, maxLength=4, stride=1, shuffle=False)
    dataIter = embedding.dataIter()
    embedding.processOneBatch(dataIter)
