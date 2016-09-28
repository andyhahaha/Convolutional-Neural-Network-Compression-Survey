# Convolution-Neural-Network-Compression-Paper

- ***XNOR-Net: ImageNet Classification Using Binary
Convolutional Neural Networks***[[arXive 2016]](https://arxiv.org/pdf/1603.05279v4.pdf) 
  - Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
  - Propose a new approach to binarize weight
  - Binarize both input and weight, and use Xnor operation to do convolution
  - Xnor-net results in 58× faster convolutional operations and 32× memory savings
  - The first work that evaluates binary neural networks on the ImageNet dataset
  - However, drop about 10% accuracy from full-precision network


- ***DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients***[[arXive 2016]](http://arxiv.org/pdf/1606.06160v2.pdf)
  - Shuchang Zhou, Zekun Ni, Xinyu Zhou, He Wen, Yuxin Wu, Yuheng Zou
  - Propose a method to train CNN with low bandwith weights, activation and gradient
  - Use bit convolution kernels to accelerate both training and inference
  - Find out that weights and activations can be deterministically quantized while gradients need to be stochastically quantized.
  - Found that that weights, activations and gradients are progressively more sensitive to bitwidth
  - Suggest that quantization bit number should be W>=1 A>=2 G>=6 which won't significantly degrade prediction accuracy

- ***Sparsifying Neural Network Connections for Face Recognition***[[arXive 2015]](https://arxiv.org/pdf/1512.01891v1.pdf)
  - Yi Sun, Xiaogang Wang, Xiaoou Tang
  - Proposes a new neural correlation-based weight selection
criterion
  - The correlation between two connected neurons are defined by the magnitude of the correlation between their neural activations
  - Claim that weight magnitude is not a good indicator of the significance
of neural connections
  - Sparsify a VGG-like network for face recognition and reach 76% compression ratio and increase 0.4% accuracy 

- ***Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding***[[ICLR 2016]](https://arxiv.org/pdf/1510.00149v5.pdf)
  - Song Han, Huizi Mao, William J. Dally
  -  Propose a three stage pipeline: pruning, trained quantization and Huffman coding
  -  Reduced the size of AlexNet by 35×, from 240MB to 6.9MB, without loss of accuracy. 
  -  Reduced the size of VGG-16 by 49×, from 552MB to 11.3MB, again with no loss of accuracy. 
- ***SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size***[[arXive 2016]](https://arxiv.org/pdf/1602.07360v3.pdf)
  - Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer 
  -  SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters
  -  With model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510× smaller than AlexNet).
  -  Strategy 1. Replace 3x3 filters with 1x1 filters
  -  Strategy 2. Decrease the number of input channels to 3x3 filters
  - Strategy 3. Downsample late in the network so that convolution layers have large activation maps
  - Propose fire module to implement above strategy

- ***Learning Structured Sparsity in Deep Neural Networks***[[NIPS 2016]](http://arxiv.org/pdf/1608.03665v3.pdf)
  - Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li
  - Claim that non-structured sparsify network cannot be sped up because of the poor cache locality and jumping memory access pattern resulted from the random pattern of the sparsity.
  - Propose a Structured Sparsity Learning (SSL) method to regularize the structures (i.e., filters, channels, filter shapes, and layer depth) of DNNs
  -  Experimental results show that SSL achieves on average 5.1× and 3.1× speedups of convolutional layer computation of AlexNet against CPU and GPU, respectively
  -  These speedups are about twice speedups of non-structured sparsity

- ***Group Sparse Regularization for Deep Neural Networks***[[arXive 2016]](https://arxiv.org/pdf/1607.00485.pdf)
  - Simone Scardapane, Danilo Comminiello, Amir Hussain, Aurelio Uncini
  - Try to optimizing (i) the weights of a deep neural network, (ii) the number of neurons for each hidden layer, and (iii) the subset of active input features "simultaneously"
  - Present "Group Lasso Penalty" to impose group-level sparsity on the network’s connections

