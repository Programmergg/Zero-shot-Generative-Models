# Zero-shot-Gnerative-Models
In this repository, we showcase a wide range of Generative Adversarial Networks (GANs) and Variational Auto-Encoder (VAEs) methods applied to the Generalized Zero-shot Learning task.

Here, we have omitted the creation of the datasets. Please refer to other Generalized Zero-shot Learning papers for dataset downloads and place them in the "datasets" folder.
Our framework is suitable for APY, AWA1, AWA2, CUB (with 312-dim attributes or 1024-dim attributes), FLO and SUN. The relevant experiments have been conducted.

First, we will showcase the results produced by various generative models under the most common GZSL training process. The specific results are as follows:
| Methods                | Source                        | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|------------------------|--------------------------------------|-----------------|---------------|---------------|
| AE                     | -                                    | 48.77%          | 56.92%        | 52.53%        |
| BIR VAE                | Bounded Information Rate Variational AutoEncoder | 53.00%          | 57.35%        | 55.09%        |
| VAE                    | -                                    | 57.87%        | 68.86%      | 62.89%        |
| BEGAN                  | BEGAN: Boundary Equilibrium GAN      | 48.74%          | 63.37%        | 55.10%        |
| DRGAN                  | DRAGAN: Deep Regret Analytic GAN     | 56.66%          | 69.24%        | 62.32%        |
| fGAN-total_variation   | f-Divergence GANs                     | 58.35%          | 67.73%        | 62.69%        |
| fGAN-forward_kl        | -                                    | 57.73%          | 72.00%  🔵    | 64.08%      |
| fGAN-reverse_kl        | -                                    | 60.11%   🔵       | 67.83%        | 63.73%        |
| fGAN-pearson           | -                                    | 57.74%          | 72.56%  🔴      | 64.18%  🔵      |
| fGAN-hellinger         | -                                    | 58.25%          | 69.64%        | 63.44%        |
| fGAN-jensen_shannon    | -                                    | 58.76%          | 69.59%        | 63.72%        |
| FisherGAN              | -                                    | 59.43%          | 67.13%        | 63.05%        |
| InfoGAN                | -                                    | 58.06%          | 70.34%        | 63.61%        |
| LSGAN                  | LSGAN: Least Squares GAN             | 54.73%          | 65.37%        | 59.58%        |
| MMGAN                  | Mini-max GAN                         | 53.59%          | 62.50%        | 57.50%        |
| NSGAN                  | Non-saturating GAN                   | 54.87%          | 63.87%        | 59.03%        |
| RaNSGAN                | Relativistic GAN                     | 46.73%          | 59.32%        | 52.28%        |
| WGAN                   | Feature Generating Networks for Zero-Shot Learning | 57.9%           | 61.4%         | 59.6%         |
| CramerGAN              | The Cramer distance as a solution to biased Wasserstein gradients | 60.62% 🔴       | 70.36%        | 65.13% 🔴     |

Note: 
1. All the results mentioned above are based on experiments conducted on AWA1, where generally generating 2,000 unseen samples yields the optimal results. However, on AWA2, it usually takes creating 4,000 unseen samples to achieve the best outcome. For CUB and SUN, typically 400-500 unseen samples are enough, and the same goes for FLO and AWA1. 
2. Although CramerGAN exhibits excellent performance, it cannot escape the phenomenon of mode collapse. If a situation arises where the Unseen Accuracy is zero, please change the value of the seed, as it can play a crucial role!

Certainly, in addition to various generative models, we also demonstrate three very useful tricks:
1. Replacing the original WGAN training with the CramerGAN training.
2. Applying a certain degree of Gaussian noise separately on seen and unseen attributes.
3. Utilizing the generative prototypes classifier.

These three tricks will significantly enhance the final results!

Under the influence of these three tricks, the current performance of GZSL is primarily represented as follows (The reason we are not considering AWA1 is that we have also tested various GZSL results with different backbones, and the AWA1 image dataset has not been open-sourced):
| Datasets | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|----------|-----------------|---------------|---------------|
| APY      | 37.1%           | 59.3%         | 45.6%         |
| FLO      | 61.5%           | 74.8%         | 67.5%         |
| AWA2     | 64.4%           | 78.7%         | 70.8%         |
| CUB      | 49.2%           | 61.6%         | 55.5%         |
| SUN      | 48.9%           | 37.9%         | 42.7%         |

Next, we will showcase the results of the GZSL task using pretrained models on ImageNet-1K and models trained from scratch on these datasets, employing backbones such as ConvNet, Vision Transformer, and Swin Transformer. The resulting outcomes are as follows:
| Dataset | Pretrained Model                   | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|---------|------------------------------------|-----------------|---------------|---------------|
| AWA2    | convnext_tiny_1k_224_ema           | 48.78% 🔴         | 64.99%   🔵     | 55.73%   🔴     |
|         | convnext_tiny_1k_224_ema_from_scratch| 37.77%        | 53.05%        | 44.13%        |
|         | convnext_small_1k_224_ema          | 45.83%    🔵      | 67.94%   🔴     | 54.73%  🔵      |
|         | convnext_small_1k_224_ema_from_scratch| 37.54%       | 50.58%        | 43.09%        |
|         | convnext_base_1k_224_ema           | 44.61%          | 64.32%        | 52.68%        |
|         | convnext_base_1k_224_ema_from_scratch | 34.98%        | 52.58%        | 42.01%        |
|         | convnext_large_1k_224_ema          | 45.23%          | 57.07%        | 50.46%        |
|         | convnext_large_1k_224_ema_from_scratch| 33.90%       | 52.05%        | 41.06%        |
| CUB     | convnext_tiny_1k_224_ema           | 33.67%     🔴     | 28.98%        | 31.15%     🔵   |
|         | convnext_tiny_1k_224_ema_from_scratch| 28.35%        | 30.79%        | 29.52%        |
|         | convnext_small_1k_224_ema          | 32.21%    🔵      | 30.62%        | 31.39%   🔴     |
|         | convnext_small_1k_224_ema_from_scratch| 27.77%       | 34.64%        | 30.83%        |
|         | convnext_base_1k_224_ema           | 29.53%          | 30.53%        | 30.02%        |
|         | convnext_base_1k_224_ema_from_scratch | 27.00%        | 36.03%  🔴      | 30.87%        |
|         | convnext_large_1k_224_ema          | 24.37%          | 35.56%    🔵    | 28.92%        |
|         | convnext_large_1k_224_ema_from_scratch| 28.19%       | 34.41%        | 30.99%        |
| CUB2    | convnext_tiny_1k_224_ema           | 27.77%          | 31.25%        | 29.41%        |
|         | convnext_tiny_1k_224_ema_from_scratch| 27.14%        | 31.81%        | 29.29%        |
|         | convnext_small_1k_224_ema          | 23.13%          | 32.46%        | 27.01%        |
|         | convnext_small_1k_224_ema_from_scratch| 33.84%  🔴     | 33.25%        | 33.54%   🔴     |
|         | convnext_base_1k_224_ema           | 23.63%          | 23.32%        | 23.48%        |
|         | convnext_base_1k_224_ema_from_scratch | 29.51%  🔵      | 35.46%  🔴      | 32.21%   🔵     |
|         | convnext_large_1k_224_ema          | 21.12%          | 21.37%        | 21.24%        |
|         | convnext_large_1k_224_ema_from_scratch| 26.29%       | 35.36%    🔵    | 30.16%        |
| SUN     | convnext_tiny_1k_224_ema           | 37.99%          | 26.74%        | 31.39%        |
|         | convnext_tiny_1k_224_ema_from_scratch| 20.00%        | 10.58%        | 13.84%        |
|         | convnext_small_1k_224_ema          | 43.40%    🔴      | 26.43%        | 32.86%  🔴      |
|         | convnext_small_1k_224_ema_from_scratch| 20.07%       | 12.64%        | 15.51%        |
|         | convnext_base_1k_224_ema           | 36.67%    🔵      | 29.22%    🔴    | 32.53%   🔵     |
|         | convnext_base_1k_224_ema_from_scratch | 21.39%        | 13.37%        | 16.46%        |
|         | convnext_large_1k_224_ema          | 35.07%          | 28.91%    🔵    | 31.70%        |
|         | convnext_large_1k_224_ema_from_scratch| 17.22%       | 12.02%        | 14.16%        |
| aPY     | convnext_tiny_1k_224_ema           | 11.43%          | 34.61%        | 17.18%        |
|         | convnext_tiny_1k_224_ema_from_scratch| 17.69%        | 36.96%        | 23.92%        |
|         | convnext_small_1k_224_ema          | 25.53%     🔵     | 15.41%        | 19.22%        |
|         | convnext_small_1k_224_ema_from_scratch| 16.91%       | 17.11%        | 17.01%        |
|         | convnext_base_1k_224_ema           | 13.51%          | 54.68%   🔴     | 21.66%   🔵     |
|         | convnext_base_1k_224_ema_from_scratch | 18.93%        | 19.03%        | 18.98%        |
|         | convnext_large_1k_224_ema          | 26.52%     🔴     | 37.23%    🔵    | 30.98%    🔴    |
|         | convnext_large_1k_224_ema_from_scratch| 17.48%       | 22.78%        | 19.78%        |
| FLO     | convnext_tiny_1k_224_ema           | 27.28%          | 10.39%        | 15.05%        |
|         | convnext_tiny_1k_224_ema_from_scratch| 50.20%   🔵     | 56.43%        | 53.13%        |
|         | convnext_small_1k_224_ema          | 21.17%          | 14.18%        | 16.99%        |
|         | convnext_small_1k_224_ema_from_scratch| 49.96%       | 66.62%        | 57.10%        |
|         | convnext_base_1k_224_ema           | 21.59%          | 19.64%        | 20.57%        |
|         | convnext_base_1k_224_ema_from_scratch | 53.90%  🔴      | 72.50%  🔵   | 61.83%   🔴     |
|         | convnext_large_1k_224_ema          | 26.01%          | 31.22%        | 28.38%        |
|         | convnext_large_1k_224_ema_from_scratch| 48.35%       | 78.45%   🔴     | 59.83%    🔵    |


| Datasets | Pretrained Model               | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|----------|--------------------------------|-----------------|---------------|---------------|
| AWA2     | ViT_base_patch16_224           | 67.81% 🔴       | 76.95% 🔵     | 72.10% 🔵   |
|          | ViT_base_patch16_224_from_scratch | 46.63%          | 61.48%         | 53.04%         |
|          | ViT_base_patch32_224           | 60.91%        | 72.74%       | 66.30%       |
|          | ViT_base_patch32_224_from_scratch | 61.14%          | 71.71%         | 66.00%         |
|          | ViT_large_patch16_224          | 66.69%  🔵        | 84.15%   🔴      | 74.41%  🔴       |
|          | ViT_large_patch16_224_from_scratch | 56.70%          | 73.21%         | 63.91%         |
| CUB      | ViT_base_patch16_224           | 55.84% 🔵       | 66.00% 🔴     | 60.50% 🔵     |
|          | ViT_base_patch16_224_from_scratch | 46.15%          | 60.45%         | 52.34%         |
|          | ViT_base_patch32_224           | 55.53%        | 62.26%     | 58.70%     |
|          | ViT_base_patch32_224_from_scratch | 51.76%          | 59.03%         | 55.15%         |
|          | ViT_large_patch16_224          | 52.54%          | 62.40%           | 57.05%         |
|          | ViT_large_patch16_224_from_scratch | 57.67% 🔴          | 65.43% 🔵        | 61.31%  🔴         |
| CUB2     | ViT_base_patch16_224           | 62.86% 🔴      | 63.03%     | 62.94%    |
|          | ViT_base_patch16_224_from_scratch | 58.26%          | 60.90%         | 59.55%         |
|          | ViT_base_patch32_224           | 59.10%          | 57.71%         | 58.40%         |
|          | ViT_base_patch32_224_from_scratch | 62.53%          | 58.26%         | 60.32%         |
|          | ViT_large_patch16_224          | 62.56% 🔵       | 64.84% 🔵     | 63.68% 🔵     |
|          | ViT_large_patch16_224_from_scratch | 69.45%          | 67.13%  🔴        | 68.27%  🔴       |
| SUN      | ViT_base_patch16_224           | 50.21%        | 49.11% 🔴     | 49.65% 🔵     |
|          | ViT_base_patch16_224_from_scratch | 37.57%          | 35.00%         | 36.24%         |
|          | ViT_base_patch32_224           | 50.28%        | 40.97%      | 45.15%    |
|          | ViT_base_patch32_224_from_scratch | 42.57%          | 41.12%         | 41.83%         |
|          | ViT_large_patch16_224          | 56.94%   🔴       | 48.68%  🔵       | 52.49%   🔴        |
|          | ViT_large_patch16_224_from_scratch | 51.67%   🔵       | 45.04%         | 48.13%         |
| aPY      | ViT_base_patch16_224           | 18.80%          | 28.53%      | 22.67%     |
|          | ViT_base_patch16_224_from_scratch | 21.93%        | 37.21% 🔴     | 27.59% 🔵     |
|          | ViT_base_patch32_224           | 20.75%          | 26.62%         | 23.32%         |
|          | ViT_base_patch32_224_from_scratch | 25.72%  🔴        | 26.86%         | 26.28%         |
|          | ViT_large_patch16_224          | 21.00%          | 32.20%         | 25.42%         |
|          | ViT_large_patch16_224_from_scratch | 24.01% 🔵         | 36.68%   🔵      | 29.03%  🔴        |
| FLO      | ViT_base_patch16_224           | 67.73% 🔵       | 56.07%         | 61.35%         |
|          | ViT_base_patch16_224_from_scratch | 70.02% 🔴       | 72.82%      | 71.39% 🔴     |
|          | ViT_base_patch32_224           | 58.66%          | 52.35%         | 55.33%         |
|          | ViT_base_patch32_224_from_scratch | 60.07%          | 75.80% 🔵     | 67.03%     |
|          | ViT_large_patch16_224          | 57.27%          | 75.59%         | 65.16%         |
|          | ViT_large_patch16_224_from_scratch | 59.96%          | 88.54%  🔴       | 71.50%  🔵        |

| Datasets | Pretrained Model                     | Unseen Accuracy | Seen Accuracy | Harmonic Mean |
|----------|--------------------------------------|-----------------|---------------|---------------|
| AWA2     | swin_tiny_patch4_window7_224         | 52.04%    🔴      | 60.55%      | 55.97%     |
|          | swin_tiny_patch4_window7_224_from_scratch | 32.68%          | 43.20%         | 37.21%         |
|          | swin_small_patch4_window7_224         | 50.46%    🔵      | 64.90%      | 56.78%     |
|          | swin_small_patch4_window7_224_from_scratch | 36.27%          | 44.70%         | 40.05%         |
|          | swin_base_patch4_window7_224          | 48.85%          | 73.64%    🔴      | 58.74%   🔴        |
|          | swin_base_patch4_window7_224_from_scratch | 33.90%          | 44.16%         | 38.35%         |
|          | swin_base_patch4_window12_384         | 48.08%          | 73.02%  🔵       | 57.98%   🔵      |
|          | swin_base_patch4_window12_384_from_scratch | 34.98%          | 45.46%         | 39.54%         |
| CUB1     | swin_tiny_patch4_window7_224         | 33.04%          | 22.39%      | 26.69%      |
|          | swin_tiny_patch4_window7_224_from_scratch | 23.47%          | 20.31%         | 21.77%         |
|          | swin_small_patch4_window7_224         | 36.56%   🔴       | 27.36%      | 31.30%      |
|          | swin_small_patch4_window7_224_from_scratch | 22.89%          | 24.62%         | 23.73%         |
|          | swin_base_patch4_window7_224          | 33.86%          | 30.61%   🔵      | 32.16%    🔵     |
|          | swin_base_patch4_window7_224_from_scratch | 22.25%          | 26.29%         | 24.10%         |
|          | swin_base_patch4_window12_384         | 34.60%  🔵        | 32.79%   🔴      | 33.67%   🔴      |
|          | swin_base_patch4_window12_384_from_scratch | 1.65%           | 1.12%          | 1.33%          |
| CUB2     | swin_tiny_patch4_window7_224         | 39.14% 🔴       | 32.32%      | 35.40% 🔴     |
|          | swin_tiny_patch4_window7_224_from_scratch | 14.19%          | 16.04%         | 15.06%         |
|          | swin_small_patch4_window7_224         | 34.60%   🔵        | 32.79% 🔵     | 33.67% 🔵     |
|          | swin_small_patch4_window7_224_from_scratch | 18.70%          | 17.40%         | 18.03%         |
|          | swin_base_patch4_window7_224          | 32.07%          | 33.97%    🔴     | 33.00%         |
|          | swin_base_patch4_window7_224_from_scratch | 20.38%          | 22.27%         | 21.28%         |
|          | swin_base_patch4_window12_384         | 33.31%          | 31.49%         | 32.38%         |
|          | swin_base_patch4_window12_384_from_scratch | 1.70%           | 0.89%          | 1.17%          |
| SUN      | swin_tiny_patch4_window7_224         | 42.78%        | 16.28%      | 23.58%     |
|          | swin_tiny_patch4_window7_224_from_scratch | 16.32%          | 14.65%         | 15.44%         |
|          | swin_small_patch4_window7_224         | 42.92%   🔵       | 23.45%      | 30.33% 🔵     |
|          | swin_small_patch4_window7_224_from_scratch | 16.18%          | 13.29%         | 14.60%         |
|          | swin_base_patch4_window7_224          | 42.99%   🔴       | 26.98% 🔵        | 33.15%    🔴     |
|          | swin_base_patch4_window7_224_from_scratch | 15.63%          | 8.49%          | 11.00%         |
|          | swin_base_patch4_window12_384         | 40.90%          | 27.87%   🔴      | 33.15%   🔴      |
|          | swin_base_patch4_window12_384_from_scratch | 1.11%           | 0.58%          | 0.76%          |
| aPY      | swin_tiny_patch4_window7_224         | 10.52%          | 27.53% 🔵     | 15.23%      |
|          | swin_tiny_patch4_window7_224_from_scratch | 14.47%          | 21.56%         | 17.32%         |
|          | swin_small_patch4_window7_224         | 17.30%          | 19.44%      | 18.31%    |
|          | swin_small_patch4_window7_224_from_scratch | 14.48%          | 26.98%         | 18.84%         |
|          | swin_base_patch4_window7_224          | 21.52%   🔵       | 20.79%         | 21.15%   🔵      |
|          | swin_base_patch4_window7_224_from_scratch | 17.51%          | 22.84%         | 19.82%         |
|          | swin_base_patch4_window12_384         | 25.60%  🔴        | 27.18%         | 26.36%  🔴       |
|          | swin_base_patch4_window12_384_from_scratch | 15.40%          | 32.54%  🔴      | 20.90%   |
