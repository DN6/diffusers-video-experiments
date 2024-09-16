import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large


class OpticalFlowProcessor:
    def preprocess(self, batch):
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            ]
        )
        batch = transforms(batch)
        return batch

    def get_optical_flow(self, frames, device):
        model = raft_large(pretrained=True, progress=False)
        model = model.to(device, torch.float32)
        model = model.eval()

        flow_maps = []
        for frame_1, frame_2 in zip(frames, frames[1:]):
            frame_1 = frame_1.to(device)
            frame_2 = frame_2.to(device)

            with torch.no_grad():
                flow_map = model(frame_1, frame_2)
                flow_maps.append(flow_map[-1].to("cpu"))

            frame_1 = frame_1.to("cpu")
            frame_2 = frame_2.to("cpu")

        model.to("cpu")
        del model
        torch.cuda.empty_cache()

        return flow_maps

    def __call__(self, frames, device):
        tensor_frames = [self.preprocess(frame)[None, :] for frame in frames]
        optical_flow_maps = self.get_optical_flow(tensor_frames, device)

        return optical_flow_maps
