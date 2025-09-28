import torch
from utils.helpers import ensure_path_join
import numpy as np


def sample_synthetic_uniform_embeddings(n_contexts):
    embeddings = torch.nn.functional.normalize(torch.randn([n_contexts, 512])).numpy()
    return {str(id_name): id_embedding for id_name, id_embedding in enumerate(embeddings)}


def sample_authentic_embeddings(n_contexts):
    all_authentic_path = "./data/test_embeddings/embeddings_resnet50_test.npy"
    all_authentic_contexts = torch.load(all_authentic_path)

    id_names = list(all_authentic_contexts.keys())[:n_contexts]
    return {id_name: all_authentic_contexts[id_name] for id_name in id_names}


if __name__ == '__main__':

    n_contexts = 10

    random_uniform_embeddings = sample_synthetic_uniform_embeddings(n_contexts)
    torch.save(random_uniform_embeddings, ensure_path_join(f"/home/moon/yy/newAID/data/contexts/syn_{n_contexts}.npy"))

    # random_authentic_embeddings = sample_authentic_embeddings(n_contexts)
    # torch.save(random_authentic_embeddings, ensure_path_join(f"data/contexts/authentic_{n_contexts}.npy")




