import torch
from scipy.stats import truncnorm

import time
import matplotlib.pyplot as plt
import torch.nn.functional as f


def generate_perturbed_contexts(context, min_cos_angle=0.9, angle_spread_factor=1):
    num, dim = context.shape

    max_perturbation_angle = torch.arccos(torch.tensor(min_cos_angle))
    angle_std_dev = max_perturbation_angle / angle_spread_factor
    upper_angle = 0
    lower_bound, upper_bound = (upper_angle - upper_angle) / angle_std_dev, \
                               (max_perturbation_angle - upper_angle) / angle_std_dev
    angle_dist = truncnorm(lower_bound.item(), upper_bound.item(), loc=upper_angle, scale=angle_std_dev.item())

    random_directions = torch.nn.functional.normalize(torch.randn(num, dim, dtype=context.dtype,
                                                                  device=context.device), dim=1)
    sample_angles = torch.tensor(angle_dist.rvs(size=num), dtype=context.dtype, device=context.device)

    cos_angles = torch.cos(sample_angles).unsqueeze(1)
    sin_angles = torch.sin(sample_angles).unsqueeze(1)

    perturbed_contexts = torch.nn.functional.normalize(cos_angles * context + sin_angles * random_directions, dim=1)

    return perturbed_contexts


if __name__ == "__main__":

    # context_example = torch.randn(1, 512)
    # context_example = torch.nn.functional.normalize(context_example)
    # context_example = context_example.repeat(1000, 1)
    #
    # perturbed_vectors = generate_perturbed_contexts(context_example)
    # cos_similarities = torch.nn.functional.cosine_similarity(perturbed_vectors, context_example)
    #
    # plt.hist(cos_similarities.numpy(), bins=30, alpha=0.75)
    # plt.title('Adjusted Cosine Similarities Distribution')
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Frequency')
    # plt.show()

    context_example = torch.randn(1, 512)
    context_example = torch.nn.functional.normalize(context_example)

    start_time = time.time()
    perturbed_context = generate_perturbed_contexts(context_example)
    end_time = time.time()

    elapsed_time = (end_time - start_time) * 1000
    print("function runtime: {:.2f} ms".format(elapsed_time))
    loss = 1-f.cosine_similarity(context_example, perturbed_context)
    print(loss.mean())
