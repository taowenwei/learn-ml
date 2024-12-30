import torch
from utils_gpt2_124M__fromOpenAI import download_and_load_gpt2
import gpt2LLM
import numpy as np


settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

gptConfig = {**gpt2LLM.GPT_CONFIG_124M}
gptConfig["qkvBias"] = True
gpt2 = gpt2LLM.GPTModel(gptConfig)
gpt2.eval()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.positionEmbeddings.weight = assign(gpt.positionEmbeddings.weight, params['wpe'])
    gpt.tokenEmbeddings.weight = assign(gpt.tokenEmbeddings.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformerBlocks[b].att.queryW.weight = assign(
            gpt.transformerBlocks[b].att.queryW.weight, q_w.T)
        gpt.transformerBlocks[b].att.keyW.weight = assign(
            gpt.transformerBlocks[b].att.keyW.weight, k_w.T)
        gpt.transformerBlocks[b].att.valueW.weight = assign(
            gpt.transformerBlocks[b].att.valueW.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformerBlocks[b].att.queryW.bias = assign(
            gpt.transformerBlocks[b].att.queryW.bias, q_b)
        gpt.transformerBlocks[b].att.keyW.bias = assign(
            gpt.transformerBlocks[b].att.keyW.bias, k_b)
        gpt.transformerBlocks[b].att.valueW.bias = assign(
            gpt.transformerBlocks[b].att.valueW.bias, v_b)

        gpt.transformerBlocks[b].att.outProjection.weight = assign(
            gpt.transformerBlocks[b].att.outProjection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformerBlocks[b].att.outProjection.bias = assign(
            gpt.transformerBlocks[b].att.outProjection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformerBlocks[b].ff.layers[0].weight = assign(
            gpt.transformerBlocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformerBlocks[b].ff.layers[0].bias = assign(
            gpt.transformerBlocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformerBlocks[b].ff.layers[2].weight = assign(
            gpt.transformerBlocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformerBlocks[b].ff.layers[2].bias = assign(
            gpt.transformerBlocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformerBlocks[b].norm1.scale = assign(
            gpt.transformerBlocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformerBlocks[b].norm1.shift = assign(
            gpt.transformerBlocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformerBlocks[b].norm2.scale = assign(
            gpt.transformerBlocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformerBlocks[b].norm2.shift = assign(
            gpt.transformerBlocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.finalNorm.scale = assign(gpt.finalNorm.scale, params["g"])
    gpt.finalNorm.shift = assign(gpt.finalNorm.shift, params["b"])
    gpt.outHead.weight = assign(gpt.outHead.weight, params["wte"])

load_weights_into_gpt(gpt2, params)
torch.save(gpt2.state_dict(), "gpt2_model.pth")