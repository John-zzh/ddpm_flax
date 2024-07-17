import jax
import jax.numpy as jnp
from flax import linen as nn
# %%
class SinusoidalPositionEmbeddings(nn.Module):
    total_time_steps: int = 1000
    time_emb_dims: int = 128
    time_emb_dims_exp: int = 512

    def setup(self) -> None:
        half_dim = self.time_emb_dims // 2

        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)

        ts = jnp.arange(self.total_time_steps, dtype=jnp.float32)

        emb = jnp.expand_dims(ts, -1) * jnp.expand_dims(emb, 0) #expand_dims 可以在指定位置插入一个新的轴，从而增加数组的维度。
        '''emb.shape = (total_time_steps, time_emb_dims // 2)'''
        # jnp.expand_dims(ts, -1) 形状为 (total_time_steps,1)
        # jnp.expand_dims(emb, 0) 形状为 (1,half_dim)

        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        '''emb.shape = (total_time_steps, time_emb_dims )'''

        # In Flax, we typically define layers in the setup method
        self.emb = emb  # We'll handle this separately as Flax doesn't have an equivalent nn.Embedding.from_pretrained

        self.dense1 = nn.Dense(features=self.time_emb_dims_exp) 
        # nn.Dense是全连接层，操作是 y = xW + b，而不是卷积层
        #  y  = (N_target_time_steps, time_emb_dims) (time_emb_dims, time_emb_dims_exp) + b
        #     = (N_target_time_steps, time_emb_dims_exp)

        # nn.Dense以x=emb[time,:]矩阵为输入，形状为（N_target_time_steps,time_emb_dims）
        # nn.Dense自动从emb的形状判断输入特征数量，也就是 W 的第一个维度, time_emb_dims
        # 需要主动指定features为输出特征数量，也就是 W 的第二个维度, time_emb_dims_exp
        # emb形状为   (some_time_steps, time_emb_dims)
        # dense1后变为  (N_target_time_steps, time_emb_dims_exp) 

        self.activation = nn.silu  #Sigmoid Linear Unit
        self.dense2 = nn.Dense(features=self.time_emb_dims_exp)
        # y = xW + b
        #    (N_target_time_steps, time_emb_dims_exp)(time_emb_dims_exp, time_emb_dims_exp)  + b
        #    = (N_target_time_steps,time_emb_dims_exp)

    def __call__(self, time):
        # Handling embedding manually since Flax doesn't support loading pretrained embeddings directly
        # in the same way PyTorch does. This is a workaround and assumes time indices are provided correctly.
        time = time.astype(jnp.int32)
        emb = self.emb[time,:]
        # emb的形状为(total_time_steps, time_emb_dims)，我们只需要拿到其中（一个）特定time的行。
        # time可以是list，同时提取很多个时间步的embedding。

        time_embedding = self.dense1(emb)
        time_embedding = self.activation(time_embedding)
        time_embedding = self.dense2(time_embedding)
        return time_embedding
    
# ## 注意力机制

class AttentionBlock(nn.Module):
    channels: int = 64
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        assert self.channels == C
        # B, Batch是批次大小
        # print('x.shape', x.shape)
        # print('B, H, W, _', B, H, W, _)

        x = x.reshape(B, H*W, self.channels)  # Reshapefor compatibility with GroupNorm and MultiHeadDotProductAttention
        # print('x.shape', x.shape)
        # print('self.channels', self.channels)

        # Flax's GroupNorm expects inputs in the shape of [N, C, ...] so we temporarily reshape
        group_norm = nn.GroupNorm(num_groups=8)(x)
        # 1.分组: num_group将"通道数"分成若干组。例如，如果通道数为 64，分成 8 组，则每组有 8 个通道。
        # 2.归一化: 在每组内计算均值和方差，然后对每组进行归一化。注意这里不是L2 归一化（向量归一化）
        #           这里是x^= x−μ/(σ2+ϵ)**0.5,其中 ϵ 是一个小的常数，用于防止除零
        # 3.缩放和平移：使用可学习的缩放和平移参数来调整归一化后的值，即out=γ⋅norm(x)+β, γ和β在学习过程中会被优化
        # 经过组归一化后，输出的形状仍然是 [B, H*W, C]，只是每个组内的通道被归一化了。
        # print('group_norm.shape', group_norm.shape)

        # Apply MultiHeadDotProductAttention
        # Note: We need to adjust this part if we want to use masked attention or have queries/keys/values different
        mhsa = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.channels, use_bias=False)(group_norm)
        # 注意力机制大概在做这件事
        # 输入的形状为 [B, N, C] N=H*W

        # # 计算查询、键、值
        # Q = linear_layer_Q(group_norm)  # [B, N, qkv_features]
        # K = linear_layer_K(group_norm)  # [B, N, qkv_features]
        # V = linear_layer_V(group_norm)  # [B, N, qkv_features]

        # # 计算注意力权重
        # attention_scores = jnp.matmul(Q, K.transpose(-2, -1)) / jnp.sqrt(qkv_features)  # [B, N, N]
        # attention_weights = nn.softmax(attention_scores, axis=-1)  # [B, N, N]

        # # 加权求和值
        # attention_output = jnp.matmul(attention_weights, V)  # [B, N, qkv_features]

        # # 如果使用多头，将每个头的输出拼接在一起，再通过一个线性层
        # # multihead_output = linear_layer_output(attention_output)

        # Reshape and add the residual connection
        output = mhsa.reshape(B, H, W, self.channels) + x.reshape(B, H, W, self.channels)
        
        return output

# ## Unet中的最小单元，残差块

class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.1
    time_emb_dims: int = 512
    apply_attention: bool = False

    @nn.compact
    def __call__(self, x, t, train: bool = True, dropout_rng=None):
        # Group 1
        # print('x.shape', x.shape)
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.silu(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h)
        # print('h.shape', h.shape)

        # Group 2 - Time Embedding
        # Assuming `t` is already a time embedding vector
        t = nn.silu(t)
        t_emb = nn.Dense(features=self.out_channels)(t)
        # print('t_emb.shape', t_emb.shape)
        # Broadcasting time embedding across spatial dimensions

        h += t_emb.reshape(-1, 1, 1, self.out_channels)
        # 等同于 h += t_emb[:, None, None, :]

        # Group 3
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.silu(h)

        # dropout_rng = self.make_rng('dropout')  else None

        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h, rng=dropout_rng)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h)

        # Match input channels to output if necessary
        if self.in_channels != self.out_channels:
            x = nn.Conv(features=self.out_channels, kernel_size=(1, 1), strides=(1, 1))(x)
        
        # Apply attention if specified
        if self.apply_attention:
            h = AttentionBlock(channels=self.out_channels)(h)
        
        #残差block的要点在于用原有的x加上处理后的x（也就是h）
        return x + h


# ## 下采样（降维）和上采样（升维）
class DownSample(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # 使用步长为2的卷积来实现下采样
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))(x)
        return h
    
class UpSample(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # 使用jax的resize函数进行最近邻上采样
        B, H, W, C = x.shape
        upsampled_x = jax.image.resize(x, shape=(B, H*2, W*2, C), method='nearest')
        # 应用卷积
        upsampled_x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))(upsampled_x)
        return upsampled_x

# %%
class UNet(nn.Module):
    input_channels: int = 3
    output_channels: int = 3
    num_res_blocks: int = 2
    base_channels: int = 128
    base_channels_multiples: tuple = (1, 2, 4, 8)
    apply_attention: tuple = (False, False, True, False)
    dropout_rate: float = 0.1
    time_multiple: int = 4

    @nn.compact
    def __call__(self, x, t, train: bool = True, dropout_rng=None):
        time_emb_dims_exp = self.base_channels * self.time_multiple # 512
        time_emb = SinusoidalPositionEmbeddings(time_emb_dims=self.base_channels, time_emb_dims_exp=time_emb_dims_exp)(t)

        h = nn.Conv(features=self.base_channels, kernel_size=(3, 3), padding='SAME')(x)
        skips = []

        # Encoder
        in_channels = self.base_channels
        for i in range(len(self.base_channels_multiples)):
            out_channels = self.base_channels * self.base_channels_multiples[i]
            # out_channels = 128 * (1, 2, 4, 8) = (128, 256, 512, 1024)
            for _ in range(self.num_res_blocks):
                h = ResnetBlock(in_channels, 
                            out_channels, 
                            self.dropout_rate, 
                            apply_attention=self.apply_attention[i])(h, time_emb, train=train, dropout_rng=dropout_rng)
                in_channels = out_channels
            skips.append(h)
            if i < len(self.base_channels_multiples) - 1:
            # if i == 0,1,2
                h = DownSample(out_channels=in_channels)(h)

        # Bottleneck
        for _ in range(2):
            h = ResnetBlock(in_channels, 
                            in_channels, 
                            self.dropout_rate, 
                            apply_attention=True)(h, time_emb, train=train, dropout_rng=dropout_rng)

        # Decoder
        for i in reversed(range(len(self.base_channels_multiples))):
            out_channels = self.base_channels * self.base_channels_multiples[i]
            if i < len(self.base_channels_multiples) - 1:
                h = UpSample(out_channels=in_channels)(h)
            skip_connection = skips.pop()
            h = jnp.concatenate([h, skip_connection], axis=-1)
            for _ in range(self.num_res_blocks + 1):
                h = ResnetBlock(in_channels + out_channels, 
                                out_channels, 
                                self.dropout_rate, 
                                apply_attention=self.apply_attention[i])(h, time_emb, train=train, dropout_rng=dropout_rng)
                in_channels = out_channels

        # Final layer
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.silu(h)
        h = nn.Conv(features=self.output_channels, kernel_size=(3, 3), padding='SAME')(h)
        return h


