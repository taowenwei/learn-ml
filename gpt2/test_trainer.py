import torch
import gpt2LLM
import trainer


torch.manual_seed(123)


def initModel(gptConfig):
    model = gpt2LLM.GPTModel(gptConfig)
    model.modelParamsInfo()
    model.eval()
    return model


gptConfig = {**gpt2LLM.GPT_CONFIG_124M}
gptConfig["contextLength"] = 256
model = initModel(gptConfig)

trainer = trainer.Trainer(model, gptConfig)
# 5.1.3 Calculating the training and validation set losses
with torch.no_grad():
    trainLoss = trainer.calcLoss(trainer.trainLoader)
    valLoss = trainer.calcLoss(trainer.valLoader)
    print("Training loss:", trainLoss)
    print("Validation loss:", valLoss)

# 5.2 Training an LLM
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
trainer.train(optimizer,
              numEpochs=10, evalFreq=5, evalIter=5,
              startInput="Every effort moves you"
              )
