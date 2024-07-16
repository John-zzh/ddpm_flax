
import jax, flax
import jax.numpy as jnp
from jax.lax import scan
from train import model, simple_diffusion_obj, WORKING_DIR
import os 
from image_process import visualize_images

input_shape=(32,32,3)
init_params = model.init(jax.random.PRNGKey(0), jnp.ones([1, *input_shape]), jnp.ones([1]))  # 用适当的输入初始化


'''使用一个已经训练好的模型来生成数据'''
def reverse_diffusion(model, params, simple_diffusion_obj, timesteps=1000, img_shape=(32,32,3),
                      num_images=5 ):

    init_key=jax.random.PRNGKey(0)
    x = jax.random.normal(init_key, (num_images, *img_shape))

    def loop_body(val,i):
        # print('i=', i)
        # print('val=', val)
        x, key = val
        time_step = timesteps - i 
        # print('time_step', time_step)
        key, subkey = jax.random.split(key)
        # noise = jax.random.normal(subkey, x.shape) if time_step > 1 else jnp.zeros_like(x)

        def true_fun(_):
            return jax.random.normal(subkey, x.shape)
    
        def false_fun(_):
            return jnp.zeros_like(x)
        
        noise = jax.lax.cond(time_step > 1, true_fun, false_fun, None)

        beta_t = simple_diffusion_obj.beta[time_step]
        one_by_sqrt_alpha_t = simple_diffusion_obj.one_by_sqrt_alpha[time_step]
        sqrt_one_minus_alpha_cumulative_t = simple_diffusion_obj.sqrt_one_minus_alpha_cumulative[time_step]

        predicted_noise = model.apply({"params": params}, x, time_step)
        new_x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + jnp.sqrt(beta_t) * noise
        )

        return (new_x, subkey), None

    '''result, final_state = jax.lax.scan(f, init, xs, length=None)
        result:是每一步调用 f 函数后返回的输出的集合。
    '''

    # print(jnp.arange(timesteps))
    _,_ = scan(loop_body, (x, init_key), jnp.arange(timesteps))
    return x

def load_model_parameters(epoch_number, log_dir='./weights'):
    """
    Load the model parameters from a saved .flax file.
    
    Args:
    epoch_number (int): The epoch number to load the parameters for.
    log_dir (str): Directory where the .flax files are stored.
    
    Returns:
    dict: The loaded model parameters.
    """
    file_path = os.path.join(log_dir, f"{epoch_number}.flax")
    print('params file path', file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved model file found for epoch {epoch_number} at {file_path}")
    
    with open(file_path, 'rb') as f:
        params_bytes = f.read()
        # 使用模型的初始参数结构进行反序列化
        params = flax.serialization.from_bytes(init_params['params'], params_bytes)
    return params

# Specify the epoch number of the model you want to load.
epoch_to_load = 800  # Adjust this to the specific epoch you need.

# Load the model parameters.
loaded_params = load_model_parameters(epoch_to_load, log_dir=f'{WORKING_DIR}/weights')
# print('loaded_params', loaded_params)

NUM_OF_IMAGES = 10

inferred_images = reverse_diffusion(model=model, 
                                    params=loaded_params, 
                                    simple_diffusion_obj=simple_diffusion_obj,
                                    num_images=NUM_OF_IMAGES)

print('inferred_images.shape', inferred_images.shape)

visualize_images(inferred_images)