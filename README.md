# Zero-shot-Gnerative-Models
In this repository, we showcase a wide range of Generative Adversarial Networks (GANs) and Variational Auto-Encoder (VAEs) methods applied to the Generalized Zero-shot Learning task.

Here, we have omitted the creation of the datasets. Please refer to other Generalized Zero-shot Learning papers for dataset downloads and place them in the "datasets" folder.
Our framework is suitable for APY, AWA1, AWA2, CUB (with 312-dim attributes or 1024-dim attributes), FLO and SUN. The relevant experiments have been conducted.

First, we will showcase the results produced by various generative models under the most common GZSL training process. The specific results are as follows:

| Methods                | Source                        | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|------------------------|--------------------------------------|-----------------|---------------|---------------|
| AE                     | -                                    | 48.77%          | 56.92%        | 52.53%        |
| BIR VAE                | Bounded Information Rate Variational AutoEncoder | 53.00%          | 57.35%        | 55.09%        |
| VAE                    | -                                    | 57.87% 🔴       | 68.86% 🔵     | 62.89%        |
| BEGAN                  | BEGAN: Boundary Equilibrium GAN      | 48.74%          | 63.37%        | 55.10%        |
| DRGAN                  | DRAGAN: Deep Regret Analytic GAN     | 56.66%          | 69.24%        | 62.32%        |
| fGAN-total_variation   | f-Divergence GANs                     | 58.35%          | 67.73%        | 62.69%        |
| fGAN-forward_kl        | -                                    | 57.73%          | 72.00% 🔴     | 64.08% 🔵     |
| fGAN-reverse_kl        | -                                    | 60.11%          | 67.83%        | 63.73%        |
| fGAN-pearson           | -                                    | 57.74%          | 72.56%        | 64.18%        |
| fGAN-hellinger         | -                                    | 58.25%          | 69.64%        | 63.44%        |
| fGAN-jensen_shannon    | -                                    | 58.76%          | 69.59%        | 63.72%        |
| FisherGAN              | -                                    | 59.43%          | 67.13%        | 63.05%        |
| InfoGAN                | -                                    | 58.06%          | 70.34%        | 63.61%        |
| LSGAN                  | LSGAN: Least Squares GAN             | 54.73%          | 65.37%        | 59.58%        |
| MMGAN                  | Mini-max GAN                         | 53.59%          | 62.50%        | 57.50%        |
| NSGAN                  | Non-saturating GAN                   | 54.87%          | 63.87%        | 59.03%        |
| RaNSGAN                | Relativistic GAN                     | 46.73%          | 59.32%        | 52.28%        |
| WGAN                   | Feature Generating Networks for Zero-Shot Learning | 57.9%           | 61.4%         | 59.6%         |
| CramerGAN              | The Cramer distance as a solution to biased Wasserstein gradients | 60.62% 🔵       | 70.36%        | 65.13% 🔴     |

Note: 
1. All the results mentioned above are based on experiments conducted on AWA1, where generally generating 2,000 unseen samples yields the optimal results. However, on AWA2, it usually takes creating 4,000 unseen samples to achieve the best outcome. For CUB and SUN, typically 400-500 unseen samples are enough, and the same goes for FLO and AWA1. 
2. Although CramerGAN exhibits excellent performance, it cannot escape the phenomenon of mode collapse. If a situation arises where the Unseen Accuracy is zero, please change the value of the seed, as it can play a crucial role!

Certainly, in addition to various generative models, we also demonstrate three very useful tricks:
1. Replacing the original WGAN training with the CramerGAN training.
2. Applying a certain degree of Gaussian noise separately on seen and unseen attributes.
3. Utilizing the generative prototypes classifier.

These three tricks will significantly enhance the final results!

Under the influence of these three tricks, the current performance of GZSL is primarily represented as follows (The reason we are not considering AWA1 is that we have also tested various GZSL results with different backbones, and the AWA1 image dataset has not been open-sourced):
<center>

| Datasets | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|----------|-----------------|---------------|---------------|
| APY      | 37.1%           | 59.3%         | 45.6%         |
| FLO      | 61.5%           | 74.8%         | 67.5%         |
| AWA2     | 64.4%           | 78.7%         | 70.8%         |
| CUB      | 49.2%           | 61.6%         | 55.5%         |
| SUN      | 48.9%           | 37.9%         | 42.7%         |

</center>
