### 发现：
- 通过词频分析，两个字组成的高频词要比三个字组成的高频词更具有实际含义。只有针对部分特别词汇（支付宝、为什么、银行卡），三个字组成的词才有使用的意义。
- 对于两个字组成的高频词，多数还是正确的常用词，可以在建模时使用。有一些高频二字词在字面意义上并不成立，他们存在是受了其他高频词的影响（比如“花呗”和“用花”），我们可以考虑通过二次统计和屏蔽重复字的方式进行筛选。
- “怎么”、“什么”、“为什么”都属于高频词，这有利于我们区分不同问题类型，再判断相似性。
- 从高频词来看，stop words的筛除并不是非常重要

### Benchmark
- Vectorize each sentence with the top 30 bi-words and the top 30 tri-words: convert each sentence into a vector of size 60 and value of 1 represents the presence of the corresponding word in that sentence. 
- Concatenate vectors from each pair: since we have a pair of sentences for each label, we horizontally merge the two vectors into one with the size 120. 
- Build an extremely simple neural network for classification:
1. 1st layer: the input size is 120 and the output size is 64, with ReLU as the activation function
2. 2nd layer: the input size is 64 and the output size is 16, with ReLU as the activation function
3. 3rd layer: the input size is 16 and the output size is 1, with Sigmoid as the activation function

Result: 85.93%
- More concern on under-fitting; less concern on over-fitting
