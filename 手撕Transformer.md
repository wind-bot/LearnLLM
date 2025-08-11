## 手撕Transformer

### 序言

本人学习transformer过程中的记录和总结，目的是把学习过程把知识输入变成知识输出来帮助自己更深的理解transformer，介绍过程从输入到架构到输出，读者需要了解Transformer模型的架构和数据流向。

### 1.Embedding层

介绍一个词向量的概念，词向量就是模型可以接受的输出。我们需要将人类的自然语言变成模型可以理解的语言，这就是Transformer模型中的Embedding层的工作。

> 额外的知识：分词
>
> 在输入神经网络之前，我们往往会先让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token （词元）并转化成一个固定的 index。输入序列就是由一组词元组成。
>
> Tips: 词元(token)一般在NLP（自然语言处理）中来说，通常指的是一个文本序列中的最小单元，可以是单词、标点符号、数字、符号或其他类型的语言元素。通常，对于NLP任务，文本序列会被分解为一系列的tokens，以便进行分析、理解或处理。
>
> 在分词的过程中会生成一个词汇表。分词的方法有根据词频率进行分词的BPE方法和Unigram model 方法（自行搜索理解，也可以先不看）
>
> 重要的知识点是分词的过程中会产生一个词汇表。

Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表。这个嵌入向量查找表也叫做"嵌入矩阵" 。**其行数为词汇表的大小(vocab_size)**。**维度（列）为嵌入维度**，嵌入维度是一个超参数，是每个词汇的向量表示的维度，用于将离散的词汇映射到连续的低维向量空间。

Embedding 层的**输入**往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer （分词器）转化成的 index 值。

Embedding层接收到输入的序列后，会将token序列在嵌入矩阵（词表）中进行对应，一个token对应一行，这一行就是这个token的向量表示，维度为嵌入维度（embedding_dim）。然后拼接成（batch_size，seq_len，embedding_dim）的矩阵输出。

上述实现并不复杂，可以直接使用 torch 中的 Embedding 层实现：

```python
token_embeddings = nn.Embedding(vocab_size,embedding_dim)
#vocab_size是一个超参数，表示词表的大小。词表是提前训练的，模型能够识别所有词的向量表格。
```

### 2.位置编码

人类的自然语言通过Embedding层可以转换为模型理解的向量。模型中有一部分是并行计算的(后面介绍)，这样每个词与词之间的位置信息就丢失了。为了保留位置信息并且实现并行计算，可以将位置信息也加入到词元的向量表示中。

原始论文中用sin、和cos 表示位置编码，为啥可以这样表示。证明过程我也没搞懂，哈哈哈。等到需要用到的时候在重新去理解吧。这里仅把公式给出来，并附上自己的代码实现。
$$
PE(pos,2i) = sin({pos \over10000^{{2i \over d_{model}}}} ) \\
PE(pos,2i+1) = cos({pos \over10000^{{2i \over d_{model}}}} )
$$
通过这两个公式计算出每个词元的位置的向量表示（$d_{modle}$表示嵌入维度）。然后在加到 Embedding层的输出中去。

```python
import numpy as np
import torch
#公式的实现
def PositionEncoding(seq_len,d_model,n=10000):
    P=np.zeros((seq_len,d_model))
    # 这里是对每个词元计算其位置编码
    for k in range(seq_len): 
        # 这里是对一个词元的位置编码的向量表示进行计算。为什么要d_model/2 是因为，这里一次计算了两个维度 2i和2i+1
        for i in np.arange(int(d_model/2)): 
            denominator = np.power(n,2*i/d_model)
            P[k,2*i] = np.sin(k/denominator)
            P[k,2*i+1] = np.cos(k/denominator)
    retorn P
```

由此我们可以定义一个位置编码层

```python
#位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self,args):
        super(PositionalEncoding,self).__init__()
        ##生成一个位置矩阵，block size 是序列的最大长度（token），n_embd为嵌入维度
        pe = torch.zeros(args.block_size,args.n_embd) 
        # 生成位置索引（0,1,2,...,max_seq_len-1），用来表示词元token在序列中的位置，并增加一个维度 【1，max_seq_len】，增加维度是为了方便计算位置向量后写入pe
        position = torch.arrange(0,args.block._size).unsqueeze(1)
        #这个是公式中$10000^{{2i \over d_{model}}$的等价指数表示
        div_term = torch.exp(torch.arange(0,args.n_embd,2)* -(math.log(1000.0)/args.n_embd))
        #sin 和cos 函数 计算位置编码
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        # 增加一个批次维度，形状变为 [1, max_seq_len, embedding_dim]，与Embedding统一维度，方便相加。
        pe = pe.unsqueeze(0)
        #将位置编码矩阵注册为非可学习的缓冲区（不会被优化器更新）不参与梯度更新，因为位置信息是不变的
        self.register_buffer("pe",pe)
    def forword(self,x):
        #将位置编码的结果添加到Embedding结果上，requires_grad_(False)明确指定位置编码不参与梯度更新（它是固定的，非可学习参数）
        x = x + self.pe[:,:x.size(1)].requires_grad(False)
        return x
```



### 3.注意力机制

加入位置信息的向量，下一步进入编码器，一个编码器包括注意机制和前馈神经网络，并在注意力机制和前馈神经网络都有一个层归一化和残差连接。接下来意义介绍。

#### 3.1 层归一化LayerNorm

归一化常见的方法分为层归一化和批归一化，两者原理类似只是执行的维度不同，只介绍层归一化。

由于深度神经网络中每一层的输入都是上一层的输出，每一层的输出结果的分布都会改变，将改变的分布输入到下一层，下一层的参数学习会在已经改变的分布上学习，因此多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变。随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。

因此，在深度神经网络中，往往需要归一化操作，将每一层的输入都归一化成标准正态分布。

归一化的本质是将数据转换为均值为 0、方差为 1 的分布，从而让模型更容易学习。层归一化的核心是：**对单个样本在同一层的所有特征（或神经元输出）进行归一化**，不依赖于批次中的其他样本。由此可知归一化需要计算均值和方差

假设某一层的输出为一个特征向量 $(x=[x_1,x_2,...,x_d])$（d 为该层的特征维度）
$$
均值：\mu = \sum_{i=1}^dx_i \\
方差：\sigma^2={1 \over d} \sum_{i=1}^d(x_i-\mu)^2
$$
将每个特征减去均值、除以标准差，得到标准化后的特征(（加入微小值 \($\epsilon$\) 避免分母为 0，即 \($\sigma^2 + \epsilon)$)
$$
\hat{x}_i={x_i-\mu \over\sqrt{\sigma^2+\epsilon}}
$$


在进行归一化后 需要进行线性映射来缩放与偏移，来保留模型的表达能力。
$$
y = \gamma \cdot \hat{x} + \beta
$$


线性映射” 作用是**赋予模型对归一化后数据的调整能力**，避免归一化操作过度 “压制” 数据的有用特征。层归一化（LayerNorm）的核心操作是将输入数据标准化为 **均值为 0、方差为 1** 的分布。但这种 “强制标准化” 可能存在问题，归一化后的数据分布被严格限制，可能会 “抹去” 输入数据中原本有用的尺度信息（例如某些特征天然需要更大的数值范围来体现重要性），不同任务、不同层对数据分布的需求不同，固定的 “0 均值 1 方差” 未必是最优分布。

其中 \($\gamma$\)和 \($\beta$\)是可学习的参数，模型可以通过训练自动调整这两个参数，让数据分布适应任务需求.

> 若 \($\gamma > 1$\)，会放大数据的波动（增加方差）反之减小方法，初值为1 表示不改变方差
>
> 若 \($\beta \neq 0$\)，会将数据整体偏移（例如从 0 均值调整为 $\beta$ 均值）,初值设置为0表示不改变均值。

通过线性映射既保留了归一化的优点（稳定训练、加速收敛、缓解梯度消失等），又允许模型根据任务需求灵活调整数据分布，确保有用的特征不会被归一化 “破坏”。

```python
#层归一化
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
    super().__init__()
    # 线性矩阵做映射
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
    
    def forward(self, x):
    # 在统计每个样本所有维度的值，求均值和方差
    mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
    std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

#### 3.2 残差连接

残差连接就是在本层的输出加上本层的输入，一起作为下一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。由于 Transformer 模型结构较复杂、层数较深，为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h(残差连接) + self.feed_forward.forward(self.fnn_norm(h))
```



#### 3.3注意力机制

层归一化和残差连接是每层都会有的一个统一操作。那么接下来介绍重点的注意力机制。

注意力机制是源于计算机视觉领域，核心思想就是关注一张图片时，模仿人类，无需看清楚全部内容而仅将注意力集中在重点部分即可。在自然语言处理领域，往往也可以通过将重点注意力集中在一个或几个 token，从而取得更高效高质的计算效果。

注意力机制核心变量有3个Query、Key和Value。可以粗略理解为Q就是我们要关注的重点。{Key:Value}就是整个序列。我们通过Attention(Q,K,V)计算来找到我们要重点关注的部分。

给出注意力计算的公式一步一步拆解
$$
Attention(Q,K,V) = softmax({QK^T \over \sqrt{d_k}})V
$$
自然语言是词汇序列，而不是图像，那么怎么衡量序列词元的注意力呢？Q矩阵是我们想要关注的词元序列，通过{K:V}是序列中的词元序列,通过$QK^T$计算的结果称为注意力分数，根据注意力分数的大小来确定我们要重点关注那个词元。

> 原理：通过合理的训练拟合，词向量能够表征语义信息，从而让语义相近的词在向量空间中距离更近，语义较远的词在向量空间中距离更远。往往用欧式距离来衡量词向量的相似性，但同样也可以用点积来进行度量：
> $$
> v.w = \sum_iv_iw_i
> $$
> 根据词向量的定义，语义相似的两个词对应的词向量的点积应该大于0，而语义不相似的词向量点积应该小于0。
>
> 那么，我们就可以用点积来计算词之间的相似度。假设我们的 Query 为“fruit”，对应的词向量为 q ；我们的 Key 对应的词向量为 $k=[v_{apple}v_{banana}v_{chair}]$,则我们可以计算 Query 和每一个键的相似程度$x=qK^T$

$Q K^T$的结果能够反映Q中词元与每一个Key的相似程度。通过softmax()函数将其转化为和为 1 的权重。再将得到的注意力分数和值向量V做对应乘积即可。这样就能找到要关注的词了。

如果 Q 和 K 对应的维度$d_k$比较大，softmax 放缩时就非常容易受影响，使不同值之间的差异较大，从而影响梯度的稳定性。因此，要将 Q 和 K 乘积的结果做一个放缩也就是除以一个$\sqrt{d_k}$。这样就得到了注意力公式(5).

```python
#注意力机制的代码实现
def attention(query,key,value,dropout):
    #获取查询矩阵Q的维度
    d_k = query.size(-1)
    #计算注意力分数
    scores =torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
```

> 解释一下dropout。这是一种防止过拟合的机制，仅在训练过程中生效。
>
> 原理：通过在训练过程中随机 “丢弃” 一部分注意力权重（注意力分数向量中的一些值），强制模型学习更加鲁棒、不依赖特定神经元（或注意力权重）的特征，从而提升模型的泛化能力。
>
> 当 `dropout` 被应用于 `p_attn` 时，会以一定概率（如常见的 0.1 或 0.2）随机将 `p_attn` 中的部分元素置为 0，同时将未被丢弃的元素按比例放大（确保整体权重的期望不变）。
>
> 注意力权重 `p_attn` 反映了输入序列中不同位置的依赖关系。如果模型过度依赖某些特定位置的注意力权重（例如，总是关注少数几个词），可能会 “记住” 训练数据中的噪声或偶然模式，导致在测试数据上表现下降。通过随机性打破权重间的过度依赖，让模型更关注全局规律而非局部噪声，最终提升对未见过数据的适应能力。
> 也就是通过随机丢弃部分权重，模型被迫学习更全面的依赖关系（不能只依赖某几个位置），从而减少对训练数据细节的过度拟合。

#### 3.4 自注意力机制

3.3说的注意力机制是计算两个序列中寻找出一个序列的每个元素对另一个序列的每个元素的相关度，然后基于相关度进行加权，即分配注意力。而这两段序列即是我们计算过程中 Q、K、V 的来源。

在 Transformer 的 Encoder 结构中，使用的是 注意力机制的变种 —— 自注意力（self-attention，自注意力）机制。所谓自注意力，即是计算本身序列中每个token对其他tokend的注意力分布，从而得到，每个词元与其他词元的关联性。即在计算过程中，Q、K、V 都由同一个输入序列通过不同的参数矩阵计算得到。

在 Encoder 中，Q、K、V 分别是输入对参数矩阵 $W_q,W_k,W_v$ 做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。通过自注意力机制，可以找到一段文本中每一个 token 与其他所有 token 的相关关系大小，从而建模文本token之间的依赖关系。

```python
#自注意力实现--就是注意力机制的Q，K，V 来自同一个序列
attention(x,x,x)
```

#### 3.5 掩码自注意力机制

掩码注意力机制可以看作是根据上一词来做预测下一个词的。通过掩码来遮住后面的词，让模型关注前面已知的词来预测下一个词。使用注意力掩码的核心动机是让模型只能使用历史信息进行预测而不能看到未来信息。不断根据之前的 token 来预测下一个 token，直到将整个文本序列补全。因为注意力机制是并行计算所以Mask是一个上三角矩阵上三角位置的元素均为 -inf，其他位置的元素置为0。来遮住后面的token

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
#在注意力计算时，我们会将计算得到的注意力分数与这个掩码做和，再进行 Softmax 操作
scores = scores + mask[:, :seqlen, :seqlen]
#Softmax 操作，-inf 的值在经过 Softmax 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽。
scores = F.softmax(scores.float(), dim=-1)
```



#### 3.6多头注意力机制

上面介绍了如何通过Q，K，V，计算注意力。但一次注意力计算只能拟合一种相关关系，单一的注意力机制很难全面拟合语句序列里的相关关系。因此 Transformer 使用了多头注意力机制（Multi-Head Attention），即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。

如果有n个注意力头，就有n组QKV矩阵，且每组参数矩阵不同，这样就会得到多个注意力结果。最后把所有头的结果拼接到一起组成输出。

多头注意力的核心思想是：**将高维空间拆分成多个低维子空间，每个头在子空间中独立计算注意力，最后拼接结果**。这相当于让模型从 “不同角度” 关注输入，而不是在单一高维空间中强行学习所有注意力模式。

> 例子：一句话有10个token,每个token的维度是512，形成[10,512] 的一个输入。单头注意力就是在这个上面进行注意力计算，只能在这10个token中捕捉一种注意力
>
> 多头注意力，假设有8个头，会把这个输入拆分为[10,64*8],在变换为[8,10,64]，相当于每个头在10个token中64个维度上进行计算注意力，那么8个注意力头所关注的角度不一样。最后再把8个头的注意力结果拼接成[10,512]

但上述实现时空复杂度均较高，我们可以通过矩阵运算巧妙地实现并行的多头计算，其核心逻辑在于使用三个组合矩阵来代替了n个参数矩阵的组合，也就是**矩阵内积再拼接其实等同于拼接矩阵再内积**

```python
import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
   def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。（多卡时拆分）
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        # 线性层：将输入映射为Query、Key、Value（合并多头参数，避免逐个定义）
        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
         
        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
           mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
           mask = torch.triu(mask, diagonal=1)
           # 注册为模型的缓冲区
           self.register_buffer("mask", mask)
   def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        #交换维度[bs,seqlen,n_heads,head_dim] -> [bs,n_heads,seqlen,head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)


        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```





### 4.前馈神经网络

前馈神经网络简称(Feed Forward Neural Network FFN), 是一个全连接的神经网络，每层网络都和上下两层的神经元完全连接的网络结构，每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。前馈神经网络的实现是较为简单的：

```python
class MLP(nn.Module):
  '''前馈神经网络'''
  def __init__(self,dim:int,hidden_dim:int,dropout:float):
    super().__init__()
    # 定义第一层线性变换，从输入维度到隐藏维度
    self.w1 = nn.Linear(dim.hidden_dim,bias=False)
    # 定义第二层线性变换，从隐藏维度到输入维度
    self.w2 = nn.Linear(hidden_dim,dim,bias=False)
    # 定义dropout层，用于防止过拟合
    self.dropout = nn.Dropout(dropout)
    
  def forward(self,x):
    # 前向传播函数
    # 首先，输入x通过第一层线性变换和RELU激活函数
    # 最后，通过第二层线性变换和dropout层
    return self.dropout(self.w2(F.relu(self.w1(x))))
```

注意，Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。



### 5.Encoder

通过上述模块的实现，现在终于可以搭建编码器了。我们可以搭建起 Transformer 的 Encoder。Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。因此，我们可以首先实现一个 Encoder Layer：

```python
class EncoderLayer(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.attention_norm = LayerNorm(args.n_embd)
    self.attention = MultiHeadAttention(args,is_causal=False)
    self.fnn_norm = LayerNorm(args.n_embd)
    self.feed_forward=MLP(args.dim,args.dim,args.dropout)
  def forward(self,x):
    norm_x = self.attention_norm(x)
    h = x + self.attention.forward(norm_x,norm_x,norm_x)
    out = h + self.feed_forward.forward(self.fnn_norm(h))
```



然后我们搭建一个 Encoder，由 N 个 Encoder Layer 组成，在最后会加入一个 Layer Norm 实现规范化：

```python
class Encoder(nn.Module):
  def __init__(self,args):
    super(Encoder,self).__init__()
    #通过超参数n_layer来建立一个encoder 有多少层EncoderLayer，用 nn.ModuleList 将这些实例包装成一个列表，存储在 self.layers 中。
    self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
    self.norm = LayerNorm(args.n_embd)
    
  def forward(self,x):
    #"分别通过 N 层 Encoder Layer"，循环列表，逐个调用。
    for layer in self.layers:
      x = layer(x)
    return self.norm(x)
```

通过 Encoder 的输出，就是输入编码之后的结果。

### 5.Decoder

类似的，我们也可以先搭建 Decoder Layer，再将 N 个 Decoder Layer 组装为 Decoder。但是和 Encoder 不同的是，Decoder 由两个注意力层和一个前馈神经网络组成。第一个注意力层是一个**掩码自注意力层**，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个**多头注意力层，该层**将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络：

```python
class DecoderLayer(nn.Module):
  def __init__(self,args):
    super().__init__()
    # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
    self.attention_norm_1 = LayerNorm(args.n_embd)
    # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True -- 表示使用掩码注意力机制
    self.mask_attention  = MultiHeadAttention(args,is_causal=True)
    # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
    self.attention_norm_2 = LayerNorm(args.n_embd)
    self.attention = MultiHeadAttention(args,is_causal=False)
    # 第三个部分是 MLP
    self.ffn_norm = LayerNorm(args.n_embd)
    self.feed_forward = MLP(args.dim,args.dim,args.dropout)
    
  def forward(self,x,enc_out):
            # Layer Norm
    norm_x = self.attention_norm_1(x)
    # 掩码自注意力
    x = x + self.mask_attention.forward(norm_x,norm_x,norm_x)
    # 多头注意力--这里的k，v 是有encoder层的输出来的
    norm_x = self.attention_norm_2(x)
    h = x + self.attention.forward(norm_x,enc_out,enc_out)
    # 经过前馈神经网络
    h = ffn_norm(h)
    out = h + self.feed_forward.forward(self,h)
    return out
    

```

然后同样的，我们搭建一个 Decoder 块：

```python
class Decoder(nn.Module):
  def __init__(self,args):
    super(Decoder,self).__init__()
    self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
    self.norm = LayerNorm(args.n_embd)
  def forward(self,x,enc_out):
    for layer in self.layers:
      x = layer(x,enc_out):
    return self.norm(x)

```



### 6.Transformer

经过 tokenizer 映射后的输出先经过 Embedding 层和 Positional Embedding 层编码，然后进入上一节讲过的 N 个 Encoder 和 N 个 Decoder（在 Transformer 原模型中，N 取为6），最后经过一个线性层和一个 Softmax 层就得到了最终输出。

基于之前所实现过的组件，我们实现完整的 Transformer 模型：

```python
class Transformer(nn.Module):
  def __init__(self,args):
    super().__init__()
    # 必须输入词表大小（词表中包含的token数量）和 block size(层数)
    assert args.vocab_size is not None
    assert args.block_size is not None
    self.args = args
    self.transformer = nn.ModluleDict(dict(
    	wte = nn.Embedding(args.vocab_size,args.n_embd), #嵌入层
      wpe = PositionalEncoding(args),#位置编码
      drop = nn.Dropout(args.dropout),#防止过拟合
      encoder = Encoder(args),#编码器
      decoder = Decoder(args),#解码器
    ))
    # 最后的线性层，输入是 n_embd，输出是词表大小
    self.lm_head = nn.Linear(args.n_embd,args.vocab_size,bias=False)
    # 初始化所有的权重
    self.apply(self.__init_weights)
    # 查看所有参数的数量
    print("number of parameters: %.2fM"%(self,get_num_params()/1e6))
    
    '''统计所有参数数量'''
  def get_num_params(self,non_embedding = False):
    # non_embedding: 是否统计 embedding 的参数
    n_params = sum(p.numel() for p in self.parameters())
    # 如果不统计 embedding 的参数，就减去
    if non_embedding:
      n_params -= self.transformer.wte.weight.numel()
    return n_params
    
    '''初始化权重'''
  def _init_weights(self,module):
    if isinstance(module,nn.Linear):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module,nn.Embedding):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
        
  def forword(self,idx,targets=None):
    # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
    device = idx.device
    b,t = idx.size()
    assert t<= self.args.block_size, f"不能计算该序列，该序列长度为{t},最大序列长度只有{self.args.block_size}"
		# 通过 self.transformer
    # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
    print("idx",idx.size())
    tok_emb = self.transformer.wte(idx)
    print("tok_emb",tok_emb.size())
    # 然后通过位置编码，再进行 Dropout--得到模型的输入
    pos_emb = self.transformer.wpe(tok_emb) 
    x = self.transformer.drop(pos_emb)
    print("x after wpe",x.size())
    # 然后通过 Encoder
    enc_out = self.transformer.encoder(x)
    print("enc_out:" enc_out.size())
    # 再通过 Decoder
    x = self.transformer.decoder(x,enc_out)
    print("x after decoder",x.size())
    
    if targets is not None:
      # 训练阶段，如果我们给了 targets，就计算 loss
      # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
      #表示 每一行表示这个位置的token预测，每一列表示对应这一列表示的token在这个位置的分数，第 0 行的第 123 列数值，代表模型预测 “第 1 个样本的第 1 个 token 位置是词表中索引为 123 的 token” 的分数。
      #语言模型 “预测 token” 的核心任务匹配，输出每个位置对应词表中所有 token 的分数；
      logits = self.lm_head(x)
      # 再跟 targets 计算交叉熵
 loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
    else:
    # 推理阶段，我们只需要 logits，loss 为 None
    # 取 -1 是只取序列中的最后一个作为输出

      logits = self.lm_head(x[:,[-1],:])
      loss =None
    return logits,loss
    

    

```
