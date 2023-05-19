import torch
import torch.nn as nn

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device).long()  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        if self.nn_model.embedding_model == 'stack':
            if self.drop_prob != 0:
                context_mask = torch.bernoulli(torch.zeros_like(c[:, 0, 0, 0]) + self.drop_prob).to(self.device)
            else:
                context_mask = None
        else:
            c, context_mask = None, None

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, (_ts / self.n_T).float(), c=c, context_mask=context_mask))

    @torch.no_grad()
    def sample_c(self, image_size, batch_size, channels, cond=None, guide_w=1.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        size = (batch_size, channels, image_size, image_size)
        x_i = torch.randn(*size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        if self.nn_model.embedding_model is not None and self.nn_model.embedding_model == 'stack':
            if self.drop_prob != 0:
                c_i = cond
                context_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
                c_i = c_i.repeat(2, 1, 1, 1)
                context_mask = context_mask.repeat(2)
                context_mask[batch_size:] = 1.
            else:
                context_mask = None
                c_i = cond

        else:
            c_i, context_mask = None, None

        # don't drop context at test time

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(batch_size)

            if self.drop_prob != 0:
                # double batch
                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2)

            z = torch.randn(*size).to(self.device) if i > 1 else 0
            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is, c=c_i, context_mask=context_mask)

            if self.drop_prob != 0:

                eps1 = eps[:batch_size]
                eps2 = eps[batch_size:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:batch_size]
            else:
                eps1 = eps[:batch_size]
                eps = eps1
                x_i = x_i[:batch_size]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
        return x_i