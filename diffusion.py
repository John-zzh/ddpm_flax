import jax
import jax.numpy as jnp

class SimpleDiffusion:
    def __init__(self, num_diffusion_timesteps=1000, img_shape=(32,32,3)):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = jnp.sqrt(self.beta)
        self.alpha_cumulative = jnp.cumprod(self.alpha, axis=0)
        self.sqrt_alpha_cumulative = jnp.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / jnp.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = jnp.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """Linear schedule, proposed in original DDPM paper."""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return jnp.linspace(beta_start, beta_end, self.num_diffusion_timesteps, dtype=jnp.float32)

def forward_diffusion(sd: SimpleDiffusion, x0: jnp.array, timestep: int, key=jax.random.PRNGKey(0)):
    # print("forward_diffusion func: x0.shape", x0.shape)
    # print("timestep:", timestep)
    # print("sd.sqrt_alpha_cumulative.shape:", sd.sqrt_alpha_cumulative.shape)

    cumulative_alpha = sd.sqrt_alpha_cumulative[timestep]
    std_dev = sd.sqrt_one_minus_alpha_cumulative[timestep]  
    eps = jax.random.normal(key, x0.shape) # Noise
    '''sqrt_alpha_cumulative[timestep] 是按照期望的timestep取出对应的alpha'''

    sample = cumulative_alpha * x0 
    sample += std_dev * eps  # Scaled inputs * scaled noise
    # print("forward_diffusion func ended normally")
    return sample, eps  

def test_forward_diffusion():
    sd = SimpleDiffusion(num_diffusion_timesteps=1000, img_shape=(32,32,3))
    key = jax.random.PRNGKey(0)
    forward_diffusion(sd=sd, x0=jax.random.normal(key, shape=(3, 32, 32, 3)),timestep=8)
    del key
# test_forward_diffusion()