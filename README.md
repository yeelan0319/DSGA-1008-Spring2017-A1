# Semi Supervised MNIST

The codebase is forked from Jake Zhao's NYU course Spring 2017 Assignment 1 for semi-supervised learning.
The main purpose for this code is to try out:

- Unsupervised learning algorithm such as GAN, VAE etc..
- Learn how semi-supervised learning works
- See how much it improves the training results compare to data augmentation and other tricks.
- Ah! Also try pytorch, they are amazingly easy to start with!

I remained the input pipeline unchanged and created semi-supervised part on top of it.
The main logic sits in `DCGAN_mnist_pytorch.py`, the code should be pretty self-explanatory.
