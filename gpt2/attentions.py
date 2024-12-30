import torch


class SelfAttention1:
    def __init__(self, inputDims, outputDims):
        self.queryW = torch.nn.Parameter(torch.rand(
            inputDims, outputDims), requires_grad=False)
        self.keyW = torch.nn.Parameter(torch.rand(
            inputDims, outputDims), requires_grad=False)
        self.valueW = torch.nn.Parameter(torch.rand(
            inputDims, outputDims), requires_grad=False)

    def forward(self, inputs):
        # 3.4 Implementing self-attention with trainable weights
        queries = inputs @ self.queryW
        # Figure 3.14
        keys = inputs @ self.keyW
        # Figure 3.15
        attnScores = queries @ keys.T
        # Scaling by the square root of the output embedding dimension
        sqrtEmbeddingDims = keys.shape[-1]
        # Figure 3.16
        attnWeights = torch.softmax(
            attnScores / sqrtEmbeddingDims**0.5, dim=-1)
        # Figure 3.17
        values = inputs @ self.valueW
        contextVector = attnWeights @ values

        # 3.5 Hiding future words with causal attention
        masked = torch.tril(torch.ones(
            attnScores.shape[0], attnScores.shape[0]))
        maskedWeight = attnWeights * masked
        rowSums = masked.sum(dim=-1, keepdim=True)
        normalizeWeight = maskedWeight / rowSums
        # 3.5.2 Masking additional attention weights with dropout
        dropout = torch.nn.Dropout(0.5)
        normalizeWeight = dropout(normalizeWeight)
        casualContextVector = normalizeWeight @ values

        return contextVector, casualContextVector


# Listing 3.3 A compact causal attention class
# contextLength: embedded token numbers in an input batch from chapter 2=
class CausalAttention(torch.nn.Module):
    def __init__(self, inputDims, outputDims, contextLength,
                 dropout, qkvBias=False):
        super().__init__()
        self.outputDims = outputDims
        self.queryW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.keyW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.valueW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(contextLength, contextLength),
                       diagonal=1)
        )

    def forward(self, inputs):
        batchSize, numTokens, inputDims = inputs.shape
        keys = self.keyW(inputs)
        queries = self.queryW(inputs)
        values = self.valueW(inputs)

        attnScores = queries @ keys.transpose(1, 2)
        attnScores.masked_fill_(
            self.mask.bool()[:numTokens, :numTokens], -torch.inf)
        attnWeights = torch.softmax(
            attnScores / keys.shape[-1]**0.5, dim=-1
        )
        attnWeights = self.dropout(attnWeights)

        casualContextVector = attnWeights @ values
        return casualContextVector


# 3.6 Extending single-head attention to multi-head attention
class MultiHeadAttention1(torch.nn.Module):
    def __init__(self, inputDims, outputDims, contextLengt,
                 dropout, numHeads, qkvBias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(
                inputDims, outputDims, contextLengt, dropout, qkvBias
            )
                for _ in range(numHeads)]
        )

    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


# 3.6.2 Implementing multi-head attention with weight splits = more efficient than MultiHeadAttention1
class MultiHeadAttention2(torch.nn.Module):
    def __init__(self, inputDims, outputDims,
                 contextLength, dropout, numHeads, qkvBias=False):
        super().__init__()
        assert (outputDims % numHeads == 0), \
            "outputDims must be divisible by numHeads"

        self.outputDims = outputDims
        self.numHeads = numHeads
        self.headDims = outputDims // numHeads
        self.queryW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.keyW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.valueW = torch.nn.Linear(inputDims, outputDims, bias=qkvBias)
        self.outProjection = torch.nn.Linear(outputDims, outputDims)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(contextLength, contextLength),
                       diagonal=1)
        )

    def forward(self, inputs):
        batchSize, numTokens, inputDims = inputs.shape
        keys = self.keyW(inputs)
        queries = self.queryW(inputs)
        values = self.valueW(inputs)

        keys = keys.view(batchSize, numTokens, self.numHeads, self.headDims)
        values = values.view(batchSize, numTokens, self.numHeads, self.headDims)
        queries = queries.view(
            batchSize, numTokens, self.numHeads, self.headDims
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attnScores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:numTokens, :numTokens]
        attnScores.masked_fill_(mask, -torch.inf)

        attnWeights = torch.softmax(
            attnScores / keys.shape[-1]**0.5, dim=-1)
        attnWeights = self.dropout(attnWeights)

        contextVector = (attnWeights @ values).transpose(1, 2)
        contextVector = contextVector.contiguous().view(
            batchSize, numTokens, self.outputDims
        )
        contextVector = self.outProjection(contextVector)
        return contextVector
