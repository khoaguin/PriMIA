import syft as sy
import torch

print(f'torch=={torch.__version__}')
print(f'syft is locate at {sy.__file__}')

use_cuda = torch.cuda.is_available()
device = torch.device(
    "cuda" if use_cuda else "cpu"
)
# weights_path = ''
# state = torch.load(weights_path, map_location=device)

hook = sy.TorchHook(torch)