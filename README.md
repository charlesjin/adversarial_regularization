# Manifold Regularization for Adversarial Robustness
A method based on manifold regularization for training adversarially robust neural networks.

https://arxiv.org/abs/2003.04286

## What's in this repo?
Currently this repo contains (1) the Resnet model used in the paper (2) pretrained weights (3) and a stub.py file for loading the model and evaluating clean accuracy.

The pretrained weights should achieve a clean accuracy of 90.84%. We also report adversarial accuracy of 71.22% using a 200-step PGD adversary with 10 random restarts.

We ran our experiments using PyTorch 1.4.0.

## Evaluation
If you have an attack that you want to submit here, feel free to send us your adversarial examples using `numpy.save`. We will update this section after verifying the results. 

Attacks need not be limited to the `l_inf` ball, though you will need to describe the attack model if not. We would be happy to link to your implementation or paper.

For now you can direct your correspondance to ccj@mit.edu.

| Adversary | Attack Model | Submitted by | Accuracy | Date |
| --------- | ------------ | ------------ | -------- | ---- |
| 200-step PGD with 10 random restarts | `l_inf (eps=8)` | initial | 71.22% | 11 Mar 2020 |
| 20-step PGD with 10 random restarts  | `l_inf (eps=8)` | initial | 72.13% | 11 Mar 2020 |

