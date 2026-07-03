import torch

from vitquant.eval.metrics import AccuracyMeter, topk_correct


def test_topk_correct():
    logits = torch.tensor([[0.1, 0.9, 0.0],   # pred 1, target 1: top1 hit
                           [0.8, 0.1, 0.1],   # pred 0, target 2: top1 miss, top2 miss
                           [0.4, 0.5, 0.1]])  # pred 1, target 0: top1 miss, top2 hit
    counts = topk_correct(logits, torch.tensor([1, 2, 0]), ks=(1, 2))
    assert counts == {1: 1, 2: 2}


def test_accuracy_meter_accumulates():
    m = AccuracyMeter(ks=(1, 5))
    logits = torch.zeros(4, 10)
    logits[torch.arange(4), torch.tensor([0, 1, 2, 3])] = 1.0
    m.update(logits, torch.tensor([0, 1, 2, 9]))  # 3/4 top1
    m.update(logits, torch.tensor([0, 1, 2, 3]))  # 4/4 top1
    assert abs(m.top1 - 7 / 8) < 1e-9
    assert m.total == 8
