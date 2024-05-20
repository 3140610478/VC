SAMPLE_RATE = 22050
N_FFT = 4096
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 256

original_train_data = "./Data/original/training"
preprocessed_train_data = "./Data/preprocessed/training"
original_eval_data = "./Data/original/evaluation"
preprocessed_eval_data = "./Data/preprocessed/evaluation"
output_eval_data = "./Data/output/evaluation"

save = "./Save"

speaker_ids = ["Trump", "Ikura",]

temporal_stride = 128

batch_size = 1

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Load = False
Gpath = "./Save/G50.model"
Dpath = "./Save/D50.model"