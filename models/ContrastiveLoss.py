import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, device, temperature: float) -> None:
        super().__init__()
        """ Initialize the Contrastive Loss module.

                Args:
                    device (torch.device): The device tensors will be transferred to (CPU or GPU).
                    temperature (float): A temperature scaling factor to apply to the logits.
        """
        self.device = device
        self.loss_computer = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, em1, em2):
        """ Forward pass of the contrastive loss calculation.
                Args:
                    em1 (Tensor): Embeddings from the first set.
                    em2 (Tensor): Embeddings from the second set.
                    em1.shape[0] <= em2.shape[0]
                Returns:
                    Tensor: The computed loss as a scalar.
        """
        # Calculate the sizes of two sets of embeddings
        m = em1.shape[0]
        n = em2.shape[0]
        if m > n:
            em1, em2 = em2, em1
            m, n = n, m

        # Normalize embeddings and compute the similarity matrix
        emb = torch.nn.functional.normalize(torch.cat([em1, em2]))
        similarity_matrix = torch.matmul(emb, emb.t()).to(self.device)

        # Create a mask for positive samples
        positives_mask = np.eye(m + n, k=n) + np.eye(m + n, k=-n)  # 将对角线向上/下偏移 n 个位置（双向）
        positives_mask = torch.from_numpy(positives_mask).to(self.device)
        positives = similarity_matrix[positives_mask.type(torch.bool)].view(m * 2, -1)  # [2m, 1]

        # Create a mask for negative samples
        negatives_mask = 1 - positives_mask - torch.from_numpy(np.eye(m + n)).to(self.device)  # 不能把自己当作负样本
        negatives_mask[m:n, :] = 0  # 因为只考虑了前 m 个（em1）和后 m 个（em2）之间的配对，中间 [m:n) 这一块行可能对应的是多出来的 em2 的尾部（当 m < n 时会出现）。这些行对应的 anchor 没有正样本对，因此在 loss 中不能算进去。
        negatives_mask = negatives_mask.type(torch.bool).to(self.device)
        negatives = similarity_matrix[negatives_mask].view(m * 2, -1)  # [2m, n + m - 2]

        # Combine positives and negatives and scale by temperature
        logits = torch.cat((positives, negatives), dim=1) / self.temperature

        # Create labels for the positive samples; all zeros because positives are always first
        labels = torch.zeros(m * 2).long().to(self.device)  # 正样本在 logits 的第一列

        # Calculate the contrastive loss
        loss = self.loss_computer(logits, labels)

        return loss

