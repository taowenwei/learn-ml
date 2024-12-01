import torch
from embedding import TokenDataLoader
import tiktoken


class Trainer:
    def __init__(self, model, gptConfig):
        self.model = model
        tokenizer = tiktoken.get_encoding("gpt2")
        with open("theVerdict.txt", "r", encoding="utf-8") as file:
            textData = file.read()
            print("Characters:", len(textData))
            print("Tokens:", len(tokenizer.encode(textData)))

            trainRatio = 0.90
            splitIdx = int(trainRatio * len(textData))
            trainData = textData[:splitIdx]
            valData = textData[splitIdx:]

            self.trainLoader = TokenDataLoader(
                trainData,
                batchSize=2,
                maxLength=gptConfig["contextLength"],
                stride=gptConfig["contextLength"],
                dropLast=True,
                shuffle=True,
                numWorkers=0
            )
            self.valLoader = TokenDataLoader(
                valData,
                batchSize=2,
                maxLength=gptConfig["contextLength"],
                stride=gptConfig["contextLength"],
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

    def computeInitLoss(self):
        def calcLoss(dataLoader, numBatches=None):
            totalLoss = 0.
            if len(dataLoader) == 0:
                return float("nan")
            elif numBatches is None:
                numBatches = len(dataLoader)  # 1
            else:
                numBatches = min(numBatches, len(dataLoader))  # 2
            for i, (inputBatch, targetBatch) in enumerate(dataLoader):
                if i < numBatches:
                    loss = self.calcBatchLoss(
                        inputBatch, targetBatch
                    )
                    totalLoss += loss.item()  # 3
                else:
                    break
            return totalLoss / numBatches  # 4

        with torch.no_grad():
            trainLoss = calcLoss(self.trainLoader)
            valLoss = calcLoss(self.valLoader)
            print("Training loss:", trainLoss)
            print("Validation loss:", valLoss)
