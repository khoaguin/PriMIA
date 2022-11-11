"""Encrypted inference using websockets and http protocol"""
import syft as sy
import torch
from pathlib import Path
import albumentations as a
from tqdm import tqdm

from torchlib.run_websocket_server import read_websocket_config
from torchlib.dataloader import (
    AlbumentationsTorchTransform,
    PathDataset,
    RemoteTensorDataset,
    CombinedLoader,
)
from torchlib.models import resnet18

print(f'\ntorch=={torch.__version__}')
print(f'syft is locate at {sy.__file__}')

PROJECT_PATH = Path(__file__).resolve().parent
print(f'project path = {PROJECT_PATH}\n')
WEIGHT_PATH = PROJECT_PATH / 'model_weights' / 'final_federated_dataserver_simulation_2022-11-08_22-15-14.pt'

cmd_args = {}
cmd_args["http_protocol"] = True

use_cuda = torch.cuda.is_available()
device = torch.device(
    "cuda" if use_cuda else "cpu"
)
state = torch.load(WEIGHT_PATH, map_location=device)

hook = sy.TorchHook(torch)

# get the configurations for the parties in the encrypted inference protocol
worker_dict = read_websocket_config(PROJECT_PATH / 'configs' / 'websetting' / 'config_inference.csv')
accessible_dict = dict()
for key, value in worker_dict.items():
    accessible_dict[value["id"]] = value
worker_dict = accessible_dict
worker_names = [name for name in worker_dict.keys()]
print(worker_dict)

# constructing the parties
data_owner = sy.grid.clients.data_centric_fl_client.DataCentricFLClient(
    hook,
    "{:s}://{:s}:{:s}".format(
        "http" if cmd_args["http_protocol"] else "ws",
        worker_dict["data_owner"]["host"],
        worker_dict["data_owner"]["port"],
    ),
    timeout=60000,
    http_protocol=cmd_args["http_protocol"],
)

crypto_provider = (
    sy.grid.clients.data_centric_fl_client.DataCentricFLClient(
        hook,
        "{:s}://{:s}:{:s}".format(
            "http" if cmd_args["http_protocol"] else "ws",
            worker_dict["crypto_provider"]["host"],
            worker_dict["crypto_provider"]["port"],
        ),
        timeout=60000,
        http_protocol=cmd_args["http_protocol"],
    )
)

model_owner = (
    sy.grid.clients.data_centric_fl_client.DataCentricFLClient(
        hook,
        "{:s}://{:s}:{:s}".format(
            "http" if cmd_args["http_protocol"] else "ws",
            worker_dict["model_owner"]["host"],
            worker_dict["model_owner"]["port"],
        ),
        timeout=60000,
        http_protocol=cmd_args["http_protocol"],
    )
)

workers = [model_owner, data_owner]
sy.local_worker.clients = [model_owner, data_owner]

args = state['args']
val_mean_std = state['val_mean_std']
mean, std = val_mean_std
num_classes = 3
tf = [
    a.Resize(args.inference_resolution, args.inference_resolution),
    a.CenterCrop(args.inference_resolution, args.inference_resolution),
]
if hasattr(args, "clahe") and args.clahe:
    tf.append(a.CLAHE(always_apply=True, clip_limit=(1, 1)))
tf.extend(
    [
        a.ToFloat(max_value=255.0),
        a.Normalize(
            mean.cpu().numpy()[None, None, :],
            std.cpu().numpy()[None, None, :],
            max_pixel_value=1.0,
        ),
    ]
)
tf = AlbumentationsTorchTransform(a.Compose(tf))
class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}

loader = CombinedLoader()
grid = sy.PrivateGridNetwork(data_owner, crypto_provider, model_owner)
data_tensor = grid.search("#inference_data")["data_owner"][0]
dataset = RemoteTensorDataset(data_tensor)
sy.local_worker.object_store.garbage_delay = 1

model = resnet18(
    pretrained=args.pretrained,
    num_classes=num_classes,
    in_channels=3 if args.pretrained else 1,
    adptpool=False,
    input_size=args.inference_resolution,
    pooling=args.pooling_type if hasattr(args, "pooling_type") else "avg",
)
model.load_state_dict(state["model_state_dict"])
model.to(device)
fix_prec_kwargs = {"precision_fractional": 16, "dtype": "long"}
share_kwargs = {
    "crypto_provider": crypto_provider,
    "protocol": "fss",
    "requires_grad": False,
}
model.fix_precision(**fix_prec_kwargs).share(*workers, **share_kwargs)
# test method
model.eval()
model.pool, model.relu = model.relu, model.pool
total_pred, total_target, total_scores = [], [], []
with torch.no_grad():
    for i, data in tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc="performing inference",
        leave=False,
    ):
        if len(data.shape) > 4:
            data = data.squeeze()
            if len(data.shape) > 4:
                raise ValueError("need 4 dimensional tensor")
        while len(data.shape) < 4:
            data = data.unsqueeze(0)
        data = data.to(device)
        data = (
            data.fix_precision(**fix_prec_kwargs)
            .share(*workers, **share_kwargs)
            .get()
        )
        # data = data.copy().get()
        output = model(data)
        output = output.get().float_prec()
        pred = output.argmax(dim=1)

        debug = 1

debug = 1