import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import flax
import os
from tqdm import tqdm
from flax.training.train_state import TrainState
import optax

from diffusion import SimpleDiffusion, forward_diffusion
from unet import UNet
from typing import Iterator
from image_process import image_generator
import argparse
import random
print(jax.devices())

WORKING_DIR="./"

def gen_args():
    def str2bool(str):
        if str.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif str.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('--epoch',       type=int,   default=3,  help='number of epochs')
    parser.add_argument('--batch_size',  type=int,   default=50,  help='batch_size')
    parser.add_argument('--retrain',  type=int,   default=None,  help='which epoch flax file')
    parser.add_argument('--learning_rate',  type=float,   default=0.001,  help='learning rate')

    args = parser.parse_args()
    return args

args =  gen_args()

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8 
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (32, 32, 3) 
    # NUM_EPOCHS = 800
    NUM_EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    LEARN_RATE = args.learning_rate
    NUM_WORKERS = 2



simple_diffusion_obj = SimpleDiffusion(num_diffusion_timesteps=1000, img_shape=(32,32,3))

class MeanMetric:
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss, n=1):
        self.total_loss += loss * n
        self.count += n

    def compute(self):
        return self.total_loss / self.count if self.count else 0

    def reset(self):
        self.total_loss = 0.0
        self.count = 0




@jax.jit
def mse_loss(pred, true):
    return jnp.mean((pred - true) ** 2)

# @jax.jit
def compute_loss(params, state, xts, gt_noise, ts, dropout_rng):
    # 生成随机key
    # keys = jax.random.split(dropout_rng, num=x0s.shape[0])
    # 应用矢量化的forward_diffusion
    # xts, gt_noise = vmap_forward_diffusion(simple_diffusion_obj, x0s, ts, keys)
    
    pred_noise = state.apply_fn({'params': params, 'rngs': {'dropout': dropout_rng}}, xts, ts)
    loss = mse_loss(gt_noise, pred_noise)
    return loss

def train_one_epoch(simple_diffusion_obj: SimpleDiffusion, loader: Iterator[np.ndarray], total_time_steps: int, state: TrainState, batches_per_epoch: int) -> tuple[TrainState, float]:
    loss_record = MeanMetric()
    rng = jax.random.PRNGKey(0)
    # 矢量化forward_diffusion
    vmap_forward_diffusion = jax.vmap(forward_diffusion, in_axes=(None, 0, 0, 0))

    # 创建tqdm进度条
    pbar = tqdm(range(batches_per_epoch), desc='Processing Batches')

    for batch in pbar:
        x0s = next(loader)
        rng, loop_rng  = jax.random.split(rng)
        ts = jax.random.randint(loop_rng, (x0s.shape[0],), 1, total_time_steps)

        '''对损失函数进行自动微分，计算关于损失梯度。'''
        dropout_rng = jax.random.split(rng, 2)[0] 

        keys = jax.random.split(dropout_rng, num=x0s.shape[0])
        # 应用矢量化的forward_diffusion
        xts, gt_noise = vmap_forward_diffusion(simple_diffusion_obj, x0s, ts, keys)

        grad_fn = jax.value_and_grad(compute_loss)

        loss, grads = grad_fn(state.params, state, xts, gt_noise, ts, dropout_rng)
        state = state.apply_gradients(grads=grads)
        loss_record.update(loss)
        pbar.set_postfix({'loss': f'{loss:.4f}'})

    mean_loss = loss_record.compute()
    return state, mean_loss

def load_model_parameters(epoch_number, log_dir, state):
    file_path = os.path.join(log_dir, f"{epoch_number}.flax")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved weights file found for epoch {epoch_number} at {file_path}")
    
    with open(file_path, 'rb') as f:
        params_bytes = f.read()
        # 确保这里的`from_bytes`方法与你的模型框架兼容
        new_params = flax.serialization.from_bytes(state.params, params_bytes)
        # 返回更新参数后的状态
        return state.replace(params=new_params)

if __name__ == '__main__':

    unet_model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[-1],
        output_channels         = TrainingConfig.IMG_SHAPE[-1],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )

    params = unet_model.init(jax.random.PRNGKey(0), jnp.ones([1, *TrainingConfig.IMG_SHAPE]), jnp.ones([1]))['params']
    '''Unet is nn.module, need a key to init it.'''

    state = TrainState.create(apply_fn=unet_model.apply, 
                            params=params, 
                            tx=optax.adam(TrainingConfig.LEARN_RATE))

    if args.retrain != None:
        state = load_model_parameters(state=state, epoch_number=args.retrain, log_dir='./weights')

    filenames = [f for f in os.listdir(f'{WORKING_DIR}/train_set') if f.endswith('.jpg') or f.endswith('.png')]
    num_files = len(filenames)
    batches_per_epoch = num_files // TrainingConfig.BATCH_SIZE
    print('batches_per_epoch', batches_per_epoch)

    mean_loss_record = []


    log_file_path = os.path.join(WORKING_DIR, 'loss_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write('#Epoch Mean Loss\n')  

    pbar = tqdm(range(1, TrainingConfig.NUM_EPOCHS+1), desc='Training Epochs')
    for epoch in pbar:
        random.shuffle(filenames)
        dataset = image_generator(filenames, batch_size=TrainingConfig.BATCH_SIZE)
        state, mean_loss = train_one_epoch(
                                simple_diffusion_obj=simple_diffusion_obj, 
                                loader=dataset, 
                                total_time_steps=TrainingConfig.TIMESTEPS,
                                batches_per_epoch=batches_per_epoch,
                                state=state)
        mean_loss_record.append(mean_loss)

        pbar.set_postfix({'mean_loss': f'{mean_loss:.4f}'})

        # 写入损失到日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{epoch} {mean_loss:.4f}\n')

        
        checkpoint_dir = os.path.join(WORKING_DIR, 'checkpoint_weights')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        current_checkpoint_path = os.path.join(checkpoint_dir, f"{epoch + args.retrain}.flax")
        previous_checkpoint_path = os.path.join(checkpoint_dir, f"{epoch + args.retrain - 1}.flax")

        # 删除上一步的文件
        if os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)

        # 保存当前的checkpoint
        with open(current_checkpoint_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(state.params))

        if epoch % 50 == 0:
            save_path = os.path.join(f'{WORKING_DIR}/weights', f"{epoch + args.retrain}.flax")
            with open(save_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(state.params))






