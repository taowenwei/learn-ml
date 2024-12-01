import torch
import gpt2LLM
import trainer


torch.manual_seed(123)

gptConfig = {**gpt2LLM.GPT_CONFIG_124M}
gptConfig["contextLength"] = 256


def initModel(gptConfig):
    model = gpt2LLM.GPTModel(gptConfig)
    model.modelParamsInfo()
    model.eval()
    return model


model = initModel(gptConfig)
trainer = trainer.Trainer(model, gptConfig)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
trainer.train(optimizer,
              numEpochs=10, evalFreq=5, evalIter=5,
              startInput="Every effort moves you"
              )
