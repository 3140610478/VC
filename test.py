import os
import sys
import torch
from torch import nn
import torchaudio

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Networks import Generator, Discriminator, Vocoder
    from Data import VCDataset, VCDataLoader
    from Data.preprocess import normalize_mel
    from Log import getLogger
    
G = torch.load(config.Gpath, config.device)

evalX = torch.load(os.path.abspath(os.path.join(
    base_folder, config.preprocessed_eval_data, "./Ikura/Ikura.data"
)))
evalY = torch.load(os.path.abspath(os.path.join(
    base_folder, config.preprocessed_eval_data, "./Trump/Trump.data"
)))
eval_dataset = VCDataset(evalX, evalY).to(config.device)

if __name__ == "__main__":
    with torch.no_grad():
        # spec, _ = torchaudio.load(os.path.join(config.original_eval_data, "Ikura/Ikura.wav"))
        # spec = Vocoder().forward(spec).squeeze(0)
        spec = torch.cat(tuple(eval_dataset.specX), dim=1)
        spec = spec.reshape(1, 1, *spec.shape)
        spec = torch.cat((spec, torch.ones_like(spec)), dim=1)
        spec = G["XY"](spec)
        spec = spec * eval_dataset.stdY + eval_dataset.meanY
        spec = spec.to(config.device)
        spec = Vocoder().to(config.device).inverse(spec)
        spec = spec.squeeze(1).cpu()
    output_folder = os.path.join(base_folder, config.output_eval_data)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "./output.wav")
    torchaudio.save(output_file, spec, config.SAMPLE_RATE)
    pass