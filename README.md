# Convolutional-Neural-Network-Compression-Paper


### Network Binarization 
---
- ***XNOR-Net: ImageNet Classification Using Binary
Convolutional Neural Networks*** [[ECCV 2016]](https://arxiv.org/pdf/1603.05279v4.pdf)  [[code]](https://github.com/allenai/XNOR-Net)
    - Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
    - Propose a new approach to binarize weight
    - Binarize both input and weight, and use Xnor operation to do convolution
    - Xnor-net results in 58× faster convolutional operations and 32× memory savings
    - The first work that evaluates binary neural networks on the ImageNet dataset
    - However, drop about 10% accuracy from full-precision network



- ***DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients*** [[arXive 2016]](http://arxiv.org/pdf/1606.06160v2.pdf)  [[code]](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/DoReFa-Net)
  - Shuchang Zhou, Zekun Ni, Xinyu Zhou, He Wen, Yuxin Wu, Yuheng Zou
  - Propose a method to train CNN with low bandwith weights, activation and gradient
  - Use bit convolution kernels to accelerate both training and inference
  - Find out that weights and activations can be deterministically quantized while gradients need to be stochastically quantized.
  - Found that that weights, activations and gradients are progressively more sensitive to bitwidth
  - Suggest that quantization bit number should be W>=1 A>=2 G>=6 which won't significantly degrade prediction accuracy

### Network Pruning & Sparsifying 
---
- ***Sparsifying Neural Network Connections for Face Recognition*** [[arXive 2015]](https://arxiv.org/pdf/1512.01891v1.pdf)
  - Yi Sun, Xiaogang Wang, Xiaoou Tang
  - Proposes a new neural correlation-based weight selection
criterion
  - The correlation between two connected neurons are defined by the magnitude of the correlation between their neural activations
  - Claim that weight magnitude is not a good indicator of the significance
of neural connections
  - Sparsify a VGG-like network for face recognition and reach 76% compression ratio and increase 0.4% accuracy 

- ***Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*** [[ICLR 2016]](https://arxiv.org/pdf/1510.00149v5.pdf) [[code]](https://github.com/songhan/Deep-Compression-AlexNet)
  - Song Han, Huizi Mao, William J. Dally
  -  Propose a three stage pipeline: pruning, trained quantization and Huffman coding
  -  Reduced the size of AlexNet by 35×, from 240MB to 6.9MB, without loss of accuracy. 
  -  Reduced the size of VGG-16 by 49×, from 552MB to 11.3MB, again with no loss of accuracy. 

- ***Learning Structured Sparsity in Deep Neural Networks*** [[NIPS 2016]](http://arxiv.org/pdf/1608.03665v3.pdf) [[code]](https://github.com/wenwei202/caffe/tree/scnn)
  - Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li
  - Claim that non-structured sparsify network cannot be sped up because of the poor cache locality and jumping memory access pattern resulted from the random pattern of the sparsity.
  - Propose a Structured Sparsity Learning (SSL) method to regularize the structures (i.e., filters, channels, filter shapes, and layer depth) of DNNs
  -  Experimental results show that SSL achieves on average 5.1× and 3.1× speedups of convolutional layer computation of AlexNet against CPU and GPU, respectively
  -  These speedups are about twice speedups of non-structured sparsity

- ***Group Sparse Regularization for Deep Neural Networks*** [[arXive 2016]](https://arxiv.org/pdf/1607.00485.pdf)
  - Simone Scardapane, Danilo Comminiello, Amir Hussain, Aurelio Uncini
  - Try to optimizing (i) the weights of a deep neural network, (ii) the number of neurons for each hidden layer, and (iii) the subset of active input features "simultaneously"
  - Present "Group Lasso Penalty" to impose group-level sparsity on the network’s connections

- ***Deep Networks with Stochastic Depth*** [[arXive 2016]](https://arxiv.org/pdf/1603.09382v2.pdf) [[code]](https://github.com/yueatsprograms/Stochastic_Depth)
  - Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
  - Stochastic depth can save training time substantially without compromising accuracy
  - Use similar concept as dropout but layer-wise
  - Stochasticly drop layer when training, but use full network when testing
  - It is kind of Implicit model ensemble

### Conditional Computation
---
- ***BranchyNet: Fast Inference via Early Exiting from
Deep Neural Networks*** [[ICPR 2016]](http://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf) [[code]](https://gitlab.com/htkung/branchynet/container_registry)
    - Surat Teerapittayanon, Bradley McDanel, H.T. Kung
    - Fast Inference with Early Exit Branches
    - Regularization via Joint Optimization
    - Mitigation of Vanishing Gradients
    - The policy of telling whether one image is inferred with high
confidence can only apply on classification task.

- ***Dynamic Deep Neural Networks:
Optimizing Accuracy-Efficiency Trade-offs by Selective Execution*** [[arXive 2017]](https://arxiv.org/pdf/1701.00299.pdf)
    - Lanlan Liu, Jia Deng
    - Add reinforcement learning agent as controller to decide which part of network should be executed.
    - Both regular modules and controller modules are learnable and are jointly trained
to optimize both accuracy and efficiency.

- ***Spatially Adaptive Computation Time for Residual Networks*** [[arXice 2016]](https://arxiv.org/abs/1612.02297)
    - Michael Figurnov, axwell D. Collins, Yukun Zhu, Li Zhang, Jonathan Huang, Dmitry Vetrov, Ruslan Salakhutdinov
    - A deep learning architecture based on Residual Network that dynamically adjusts the number of executed layers for the regions of the image
    -  It is well-suited for a wide range of computer vision problems, including multi-output and per-pixel prediction problems.
    - We evaluate the computation time maps on the visual saliency dataset cat2000 and find that they correlate surprisingly well with human eye fixation positions.

### Network Distilling 
---
- ***Distilling the Knowledge in a Neural Network*** [[arXive 2015]](https://arxiv.org/pdf/1503.02531v1.pdf)
    - Geoffrey Hinton, Oriol Vinyals, Jeff Dean
    - Pre-train a large model as teacher and "distill" the knowledge to a small model that is more suitable for deployment
    - An important way to transfer the generalization ability of the teacher model to student model is to use the class probabilities produced by the cumbersome model as “soft targets” for training the small model rather than only the one-hot vector of the teacher's decision.

- ***FITNETS: HINTS FOR THIN DEEP NETS*** [[ICLR 2015]](https://arxiv.org/pdf/1412.6550.pdf) [[code]](https://github.com/adri-romsor/FitNets)
    - Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio

- ***Do Deep Nets Really Need to be Deep?*** [[NIPS 2014]](https://arxiv.org/pdf/1312.6184.pdf) 
    - Lei Jimmy Ba, Rich Caruana

- ***Deep Model Compression: Distilling Knowledge from Noisy Teachers*** [[arXive 2016]](https://arxiv.org/pdf/1610.09650.pdf)
    - Bharat Bhusan Sau, Vineeth N. Balasubramanian

### Designing compact layers 
---
- ***SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*** [[arXive 2016]](https://arxiv.org/pdf/1602.07360v3.pdf) [[code]](https://github.com/DeepScale/SqueezeNet) [[code with Deep Compression]](https://github.com/songhan/SqueezeNet-Deep-Compression) 
  - Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer 
  -  SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters
  -  With model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510× smaller than AlexNet).
  -  Strategy 1. Replace 3x3 filters with 1x1 filters
  -  Strategy 2. Decrease the number of input channels to 3x3 filters
  - Strategy 3. Downsample late in the network so that convolution layers have large activation maps
  - Propose fire module to implement above strategy



