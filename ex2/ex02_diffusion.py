import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
import matplotlib.pyplot as plt

def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # 计算 alphas_cumprod (即 bar_alpha)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # 根据 bar_alpha 推导 beta
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    betas = torch.linspace(-6, 6, timesteps)
    
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    return betas

def plot_beta_schedulers(timesteps, beta_start, beta_end):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(betas.numpy())
    plt.title("Beta Schedule")
    plt.xlabel("Timestep")
    plt.ylabel("Beta")

    plt.subplot(1, 2, 2)
    plt.plot(alphas_cumprod.numpy())
    plt.title("Alpha Cumprod (Signal Retention)")
    plt.xlabel("Timestep")
    plt.ylabel("Alpha_bar")
    
    plt.show()

class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda", num_classes=None):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device
        self.num_classes = num_classes

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps).to(device) # 得到一个tensor 里面包含每个时间步的 β, 每一步要加的噪声方差
        
        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        
        # define alphas
        # TODO
        self.alpha_t = 1 - self.betas # 计算alpha
        self.alphas_cum = torch.cumprod(self.alpha_t, dim=0) # 求连乘的积 cumulative product
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.sqrt_alphas_cum = torch.sqrt(self.alphas_cum)
        self.sqrt_1m_alphas_cum = torch.sqrt(1 - self.alphas_cum)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.sqrt_alphas_inv = torch.sqrt(1 / self.alpha_t)
        self.sqrt_1m_alphas_cum_inv_beta = torch.sqrt(1 / (1 - self.alphas_cum)) * self.betas
        self.sqrt_1m_alphas_cum_inv_beta = self.sqrt_1m_alphas_cum_inv_beta.to(device)

        self.sqrt_betas = torch.sqrt(self.betas)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None, guidance_weight=0.3):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        # TODO (2.2): The method should return the image at timestep t-1.
        if t_index > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        if y is not None and self.num_classes is not None and guidance_weight > 0:
            # Conditional prediction
            eps_cond = model(x, t, y)
            # Unconditional prediction (null token)
            null_token_idx = self.num_classes  # Assuming null is at index num_classes
            null_labels = torch.full_like(y, null_token_idx)
            eps_uncond = model(x, t, class_labels=null_labels)

            eps = (1 + guidance_weight) * eps_cond - guidance_weight * eps_uncond
        else:
            # Unconditional or no guidance
            eps = model(x, t, y)

        # TODO (2.2): The method should return the image at timestep t-1.
        x_t_min = (self.sqrt_alphas_inv[t_index] *
                   (x - self.sqrt_1m_alphas_cum_inv_beta[t_index] * eps) +
                   self.sqrt_betas[t_index] * z)
        return x_t_min

        

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, guidance_weight=0.3, y=None):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.

        # TODO (2.2): Return the generated images
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)

        # 2. 迭代去噪: 从 T-1 循环到 0
        # 使用 tqdm 可以显示进度条（如果不需要可以去掉）
        # for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 将 guidance_scale 传下去
            img = self.p_sample(model, img, t, i, y=y, guidance_weight=guidance_weight)

        return img
        

        

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)
            
        # 辅助函数：从预计算的 schedule 中提取当前 t 的值并 reshape
        # 确保系数和 x_zero 在同一个 device


        sqrt_alphas_cum = extract(self.sqrt_alphas_cum, t, x_zero.shape)
        sqrt_1m_alphas_cum = extract(self.sqrt_1m_alphas_cum, t, x_zero.shape)
        
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        x_t = sqrt_alphas_cum * x_zero + sqrt_1m_alphas_cum * noise
        
        return x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1", y=None):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        # 1. 如果没有提供噪声，则生成随机噪声 (epsilon)
        if noise is None:
            noise = torch.randn_like(x_zero)

        # 2. 只有前向过程 q_sample 得到 x_t (Noisy Image)
        # Put Noise on the image
        x_t = self.q_sample(x_zero, t, noise)

        # 3. 使用模型预测噪声 (Predict noise)
        # 根据是否提供 y (条件) 来决定是否传入模型
        if y is not None:
            predicted_noise = denoise_model(x_t, t, y)
        else:
            predicted_noise = denoise_model(x_t, t)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise NotImplementedError()

        return loss
