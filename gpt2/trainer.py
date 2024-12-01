import torch
from embedding import TokenDataLoader, text2token, token2text
import tiktoken
from gpt2LLM import generateTextSimple


class Trainer:
    def __init__(self, model, gptConfig):
        self.model = model
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.contextLength = gptConfig["contextLength"]
        with open("theVerdict.txt", "r", encoding="utf-8") as file:
            textData = file.read()
            print("Characters:", len(textData))
            print("Tokens:", len(self.tokenizer.encode(textData)))

            trainRatio = 0.90
            splitIdx = int(trainRatio * len(textData))
            trainData = textData[:splitIdx]
            valData = textData[splitIdx:]

            self.trainLoader = TokenDataLoader(
                trainData,
                batchSize=2,
                maxLength=self.contextLength,
                stride=self.contextLength,
                dropLast=True,
                shuffle=True,
                numWorkers=0
            )
            self.valLoader = TokenDataLoader(
                valData,
                batchSize=2,
                maxLength=self.contextLength,
                stride=self.contextLength,
                dropLast=False,
                shuffle=False,
                numWorkers=0
            )

            print("Train loader:")
            for x, y in self.trainLoader:
                print(x.shape, y.shape)

            print("\nValidation loader:")
            for x, y in self.valLoader:
                print(x.shape, y.shape)

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def calcBatchLoss(self, inputBatch, targetBatch):
        inputBatch = inputBatch.to(self.device)  # 1
        targetBatch = targetBatch.to(self.device)
        logits = self.model(inputBatch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targetBatch.flatten()
        )
        return loss

    def calcLoss(self, dataLoader, numBatches=None):
        totalLoss = 0.
        if len(dataLoader) == 0:
            return float("nan")
        elif numBatches is None:
            numBatches = len(dataLoader)
        else:
            numBatches = min(numBatches, len(dataLoader))
        for i, (inputBatch, targetBatch) in enumerate(dataLoader):
            if i < numBatches:
                loss = self.calcBatchLoss(
                    inputBatch, targetBatch
                )
                totalLoss += loss.item()
            else:
                break
        return totalLoss / numBatches

    def evaluateModel(self,  evalIter):
        self.model.eval()
        with torch.no_grad():
            trainLoss = self.calcLoss(self.trainLoader, numBatches=evalIter)
            valLoss = self.calcLoss(self.valLoader, numBatches=evalIter)
        self.model.train()
        return trainLoss, valLoss

    def generateForEpoch(self, startInput):
        self.model.eval()
        contextSize = self.model.positionEmbeddings.weight.shape[0]
        encoded = text2token(self.tokenizer, startInput).to(self.device)
        with torch.no_grad():
            token_ids = generateTextSimple(
                model=self.model, textTokens=encoded,
                maxTokenToGenerate=50, contextLength=contextSize
            )
        decoded_text = token2text(self.tokenizer, token_ids)
        print(decoded_text.replace("\n", " "))
        self.model.train()

    def train(self, optimizer, numEpochs,
              evalFreq, evalIter, startInput):

        trainLosses, valLosses, trackTokensSeen = [], [], []
        tokensSeen, globalStep = 0, -1

        for epoch in range(numEpochs):
            self.model.train()
            for inputBatch, targetBatch in self.trainLoader:
                optimizer.zero_grad()
                loss = self.calcBatchLoss(inputBatch, targetBatch)
                loss.backward()
                optimizer.step()
                tokensSeen += inputBatch.numel()
                globalStep += 1

                if globalStep % evalFreq == 0:
                    trainLoss, valLoss = self.evaluateModel(evalIter)
                    trainLosses.append(trainLoss)
                    valLosses.append(valLoss)
                    trackTokensSeen.append(tokensSeen)
                    print(f"Ep {epoch+1} (Step {globalStep:06d}): "
                          f"Initial loss {loss:.3f}, "
                          f"Train loss {trainLoss:.3f}, "
                          f"Val loss {valLoss:.3f}"
                          )

            self.generateForEpoch(startInput)
        return trainLosses, valLosses, trackTokensSeen
