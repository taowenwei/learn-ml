import torch
from embedding import TextEmbedding

torch.manual_seed(123)

embedding = TextEmbedding(
    batchSize=1, maxLength=4, stride=1, shuffle=False)
dataIter = embedding.dataIter()
embedding.processOneBatch(dataIter)
