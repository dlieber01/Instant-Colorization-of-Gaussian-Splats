import torch
import torch.nn.functional as F

def compute_image_gradient(image: torch.Tensor) -> torch.Tensor:
    # image: (1, 3, H, W) in [0, 1]
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    grads = []
    for c in range(3):
        ch = image[:, c:c+1, :, :]
        gx = F.conv2d(ch, sobel_x, padding=1)
        gy = F.conv2d(ch, sobel_y, padding=1)
        g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        grads.append(g)
    return torch.cat(grads, dim=1)  # shape: (1, 3, H, W)
