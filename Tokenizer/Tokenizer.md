## Tokenizer

### 1. BPE字节对编码分词器

分词是对大模型的embedding层服务的。用于将用户输入的序列分词为token，并返回对应token的id。将序列转化为id输入到embedding层。然后根据id转化为对应张量。

> 对于算法的原理详细解释，请看维基百科https://en.wikipedia.org/wiki/Byte-pair_encoding



#### 算法原理（个人理解）

原始算法是用来压缩字节的，用在大语言模型上可以更好的压缩token，是大模型能理解更长的上下文。个人理解的就是让一个token尽可能包含多的信息。该算法是对字节进行操作的。

给出一个字符序列。刚开始序列中的每个字节就是一个token。例如：aaabdaaabac

> 然后从头到尾，两两合并这些token，生成一个新的 token序列，统一下新的token序列中 那个token出现的次数最多，就把它作为新的token。[aa,aa ab,bd,da,aa,aa,ab,ba,ac]--其中 aa出现的次数最多。
> 就把 aa 作为一个新的token加入到词表中。用Z代替aa修改原序列--[zabdzabac]
>
> 在从新token序列中两辆结合得到[za,ab,bd,dz,za,ab,ba,ac], 这时发现ab，和za 都出现了两次。这时候随便选一个ab，作为新的token加入词表中。在把Y=ab带入到原序列。直到达到指定词表大小。或者没有重复的token了。

每次选出一个新token来，就加入词表中并产生对应的id。



