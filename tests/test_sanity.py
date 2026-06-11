import torch


def test_environment_sane():
    # CPU tensor math works and we are NOT on CUDA (this is the Mac).
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0
    assert not torch.cuda.is_available()