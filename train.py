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

    args = parser.parse_args()
    return args

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
    LR = 2e-4
    NUM_WORKERS = 2

model = UNet(
    input_channels          = TrainingConfig.IMG_SHAPE[-1],
    output_channels         = TrainingConfig.IMG_SHAPE[-1],
    base_channels           = ModelConfig.BASE_CH,
    base_channels_multiples = ModelConfig.BASE_CH_MULT,
    apply_attention         = ModelConfig.APPLY_ATTENTION,
    dropout_rate            = ModelConfig.DROPOUT_RATE,
    time_multiple           = ModelConfig.TIME_EMB_MULT,
)

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

def create_train_state(rng, model: UNet, learning_rate, input_shape):
    params = model.init(rng, jnp.ones([1, *input_shape]), jnp.ones([1]))['params']

    '''Adam 优化器，用于更新模型参数。'''
    optimizer = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state

def mse_loss(pred, true):
    return jnp.mean((pred - true) ** 2)

def train_one_epoch(simple_diffusion_obj: SimpleDiffusion, loader: Iterator[np.ndarray], total_time_steps: int, state: TrainState, batches_per_epoch: int) -> tuple[TrainState, float]:
    loss_record = MeanMetric()
    rng = jax.random.PRNGKey(0)
    for batch in range(batches_per_epoch):
        print(f'=============== batch {batch} ================')
        x0s = next(loader)
        rng, loop_rng  = jax.random.split(rng)
        ts = jax.random.randint(loop_rng, (x0s.shape[0],), 1, total_time_steps)
        # print('x0s.shape', x0s.shape)
        # print('ts.shape', ts.shape)

        def compute_loss(params, simple_diffusion_obj, x0s, ts, dropout_rng):

            # 矢量化forward_diffusion
            vmap_forward_diffusion = jax.vmap(forward_diffusion, in_axes=(None, 0, 0, 0))
            
            # 生成随机key
            keys = jax.random.split(dropout_rng, num=x0s.shape[0])
            # 应用矢量化的forward_diffusion
            xts, gt_noise = vmap_forward_diffusion(simple_diffusion_obj, x0s, ts, keys)
            
            pred_noise = state.apply_fn({'params': params, 'rngs': {'dropout': dropout_rng}}, xts, ts)
            loss = mse_loss(gt_noise, pred_noise)
            return loss
        
        '''对损失函数进行自动微分，计算关于损失梯度。'''
        dropout_rng = jax.random.split(rng, 2)[0] 
        grad_fn = jax.value_and_grad(compute_loss)

        loss, grads = grad_fn(state.params, simple_diffusion_obj, x0s, ts, dropout_rng)
        state = state.apply_gradients(grads=grads)
        loss_record.update(loss)
        print(f'=============== loss {loss} ================')

    mean_loss = loss_record.compute()
    print('mean_loss', mean_loss)
    return state, mean_loss

def train():
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(rng=init_rng, 
                               model=model, 
                               learning_rate=0.001, # Example learning rate
                               input_shape=(32,32,3))  
    del init_rng

    num_files = 733
    batches_per_epoch = num_files // TrainingConfig.BATCH_SIZE
    print('batches_per_epoch', batches_per_epoch)
    mean_loss_record = []
    dataset = image_generator(f'{WORKING_DIR}/train_set', batch_size=TrainingConfig.BATCH_SIZE, num_files_limit=None)
    for epoch in tqdm(range(TrainingConfig.NUM_EPOCHS), desc='Training Epochs'):
        print('===================== epoch =',epoch)
        state, mean_loss = train_one_epoch(
                                simple_diffusion_obj=simple_diffusion_obj, 
                                loader=dataset, 
                                total_time_steps=TrainingConfig.TIMESTEPS,
                                batches_per_epoch=batches_per_epoch,
                                state=state)
        mean_loss_record.append(mean_loss)
        # if epoch % 100 == 0:
        if epoch:
            # 每100个epoch保存模型参数
            params = state.params
            save_path = os.path.join(f'{WORKING_DIR}/log', f"{epoch}.flax")
            with open(save_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(params))
    return state

if __name__ == '__main__':
    state = train()


