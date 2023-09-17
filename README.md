# Zero-shot-Gnerative-Models
In this repository, we showcase a wide range of Generative Adversarial Networks (GANs) and Variational Auto-Encoder (VAEs) methods applied to the Generalized Zero-shot Learning task.

Here, we have omitted the creation of the datasets. Please refer to other Generalized Zero-shot Learning papers for dataset downloads and place them in the "datasets" folder.
Our framework is suitable for APY, AWA1, AWA2, CUB (with 312-dim attributes or 1024-dim attributes), FLO and SUN. The relevant experiments have been conducted.

First, we will showcase the results produced by various generative models under the most common GZSL training process. The specific results are as follows:
| 方法名字                | 来源(如果适用)                        | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
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

