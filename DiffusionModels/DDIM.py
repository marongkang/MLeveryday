from DDPM import DDPM
from tqdm import tqdm
import torch


class DDIM(DDPM):
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self,
                        img_or_shape,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20,
                        eta=1):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            # ！！！
            # 此处参考笔记中的“加速推理”部分
            # 下文中的alpha和DDPM的alpha不一样了
            # DDIM的alpha对应的是DDPM的 alpha_bar
            alpha_cur = self.alpha_bars[cur_t]
            alpha_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)
            var = eta * (1 - alpha_prev) / (1 - alpha_cur) * (1 - alpha_cur / alpha_prev)
            noise = torch.randn_like(x)

            first_term = (alpha_prev / alpha_cur)**0.5 * x
            second_term = ((1 - alpha_prev - var)**0.5 -
                            (alpha_prev * (1 - alpha_cur) / alpha_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - alpha_cur / alpha_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x