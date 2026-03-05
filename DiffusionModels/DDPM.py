import torch


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    # x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon
    # where \bar{\alpha_t} is self.alpha_bars[t]
    #       \epsilon samples from N(0,1)
    def sample_forward(self, x, t, eps=None):
        # print("alpha_bars:", self.alpha_bars[t], t)

        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # print(alpha_bar.shape)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor) # predict \epsilon_t,noted as \epsilon_{theta}

        if t == 0:
            sample = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                        1 - self.alpha_bars[t]) * self.betas[t]
            sample = torch.randn_like(x_t)
            sample *= torch.sqrt(var)

        # \bar{\mu_t} = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\epsilon_t)
        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        x_t = mean + sample
        return x_t


# test sample_forward()
# ddpm = DDPM(device,100)
# x_0 = next(enumerate(get_dataloader(batch_size=1)))[1][0]
# x = x_0.repeat(100,1,1,1).to(device)
# t = torch.tensor(range(100)).reshape(100,1).to(device)
# res = ddpm.sample_forward(x,t).to("cpu")
# print(res.shape)
#
#
# def show_image(image, ax):
#     image = image.squeeze()  # Remove single-dimensional entries from the shape
#     ax.imshow(image, cmap='gray')
#
#
# fig, axes = plt.subplots(10, 10, figsize=(10, 7))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     show_image(res[i], ax)
#     ax.axis('off')  # Hide the axes
#
# plt.tight_layout()
# plt.show()
