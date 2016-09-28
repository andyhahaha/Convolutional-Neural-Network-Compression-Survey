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
