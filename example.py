import torch
from pig.models import PeppaPig
from pig.data import audiofile_loader
import glob

net = PeppaPig.load_from_checkpoint("lightning_logs/version_31/checkpoints/epoch=101-step=17645-v1.ckpt")
net.eval()
net.cuda()
with torch.no_grad():
    audio_paths = glob.glob(f"data/out/realign/narration/ep_1/0/*.wav")
    loader = audiofile_loader(audio_paths)
    emb = torch.cat([ net.encode_audio(batch.to(net.device)).squeeze(dim=1)
                          for batch in loader ])

print(f"Audio embedding tensor with shape: {emb.shape}")
