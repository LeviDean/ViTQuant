import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor,
                 ks: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    """Number of samples whose target is within the top-k predictions, per k."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.unsqueeze(1))
    return {k: int(correct[:, :min(k, maxk)].any(dim=1).sum()) for k in ks}


class AccuracyMeter:
    def __init__(self, ks: tuple[int, ...] = (1, 5)):
        self.ks = ks
        self.correct = {k: 0 for k in ks}
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        counts = topk_correct(logits, targets, self.ks)
        for k in self.ks:
            self.correct[k] += counts[k]
        self.total += targets.numel()

    @property
    def top1(self) -> float:
        return self.correct[1] / max(self.total, 1)

    @property
    def top5(self) -> float:
        return self.correct[5] / max(self.total, 1)
