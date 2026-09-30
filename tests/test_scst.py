"""CPU checks for scst.py on a toy model with ClipCaptionModel's forward interface.

Run with: python -m pytest tests/test_scst.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scst  # noqa: E402

V, D, L, P = 12, 16, 8, 3  # vocab, width, max caption length, image tokens


class ToyCaptioner(nn.Module):
    """Same forward signature and attributes as ClipCaptionModel, a few thousand weights."""

    def __init__(self):
        super().__init__()
        self.num_classes = V + 1
        self.time_step = L
        self.wte = nn.Embedding(V, D)
        self.bos_embedding = nn.Parameter(torch.randn(D))
        self.pad_embedding = nn.Parameter(torch.randn(P, D, dtype=torch.float64))
        self.pos = nn.Parameter(torch.randn(L, D) * 0.1)
        self.temb = nn.Embedding(L, D)
        self.clip_project = nn.Linear(D, D)
        self.len_head = nn.Linear(D, L)
        self.attn = nn.Linear(D, D)
        self.head = nn.Linear(D, V)

    def image_encode(self, image):
        return image, image.mean(1)

    def forward(self, tokens, mask_tokens, prefix, mask=None, t=None, labels=None, image_feats=None, image_free=None):
        if image_feats is None:
            prefix, len_cls = self.image_encode(prefix)
        else:
            prefix, len_cls = image_feats
        img = self.clip_project(prefix)
        if image_free is not None:
            img = torch.where(image_free[:, None, None], self.pad_embedding.unsqueeze(0).to(img.dtype), img)
        x = self.wte(tokens)
        x = torch.where((mask_tokens == self.num_classes - 1)[..., None], self.bos_embedding.expand_as(x), x)
        x = x + self.pos + self.temb(t)[:, None] + img.mean(1)[:, None]
        att = (x @ self.attn(x).transpose(1, 2)) / D ** 0.5 + mask
        x = x + att.softmax(-1) @ x
        return SimpleNamespace(logits=self.head(torch.tanh(x))), self.len_head(len_cls)


@pytest.fixture(scope='module', autouse=True)
def process_group():
    init_file = tempfile.NamedTemporaryFile(delete=False)
    dist.init_process_group('gloo', init_method=f'file://{init_file.name}', rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def make(seed=0, n=6):
    torch.manual_seed(seed)
    core = ToyCaptioner()
    image = torch.randn(n, P, D)
    feats = core.image_encode(image)
    length = torch.randint(2, L + 1, (n,))
    return core, DDP(core), feats, length


def test_greedy_rollout_fills_exactly_the_predicted_length():
    core, _, feats, length = make()
    tokens, steps = scst.rollout(core, feats, length, sample=False, max_len=L)
    mask_id = core.num_classes - 1
    pos = torch.arange(L)[None]
    assert steps == []
    assert ((tokens != mask_id) == (pos < length[:, None])).all()
    again, _ = scst.rollout(core, feats, length, sample=False, max_len=L)
    assert torch.equal(tokens, again)


@pytest.mark.parametrize('guidance,sample_slot', [(1.0, False), (1.06, False), (1.0, True), (1.5, True)])
def test_replayed_log_prob_matches_rollout(guidance, sample_slot):
    core, model, feats, length = make()
    tokens, steps = scst.rollout(core, feats, length, sample=True, guidance_scale=guidance,
                                 sample_slot=sample_slot, max_len=L)
    assert len(steps) == int(length.max())
    advantage = torch.randn(length.size(0))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    _, gap = scst.policy_gradient_backward(model, core, feats, length, steps, advantage, scaler,
                                           guidance_scale=guidance, sample_slot=sample_slot,
                                           sample_length=True, replay_steps=0, chunk_size=4)
    assert gap < 1e-5
    for name, p in core.named_parameters():
        assert p.grad is not None, name


def test_full_replay_gradient_equals_autograd_of_the_trajectory_log_prob():
    core, model, feats, length = make(seed=1)
    _, steps = scst.rollout(core, feats, length, sample=True, max_len=L)
    advantage = torch.randn(length.size(0))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    scst.policy_gradient_backward(model, core, feats, length, steps, advantage, scaler,
                                  replay_steps=0, chunk_size=5)
    replay_grad = core.head.weight.grad.clone()

    # reference: -sum_j A_j sum_k log p(w_jk) / sum(length), computed step by step with autograd
    core.zero_grad()
    mask_id = core.num_classes - 1
    total = 0.0
    rows = torch.arange(length.size(0))
    for k, s in enumerate(steps):
        logp, _ = scst.policy_log_probs(core, mask_id, s['state'], feats, s['t'])
        active = length > k
        lp = logp[rows, s['slot'], s['token']] * active.float()
        total = total - (advantage * lp).sum()
    (total / length.sum()).backward()
    assert torch.allclose(replay_grad, core.head.weight.grad, atol=1e-5)


def test_policy_gradient_raises_the_reward():
    """Reward = how many times token 3 appears; SCST with a greedy baseline should learn it."""
    core, model, feats, length = make(seed=2, n=16)
    opt = torch.optim.Adam(core.parameters(), lr=3e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    n = 4
    rep_feats = tuple(f.repeat_interleave(n, 0) for f in feats)
    rep_len = length.repeat_interleave(n)

    def reward(tokens, lens):
        pos = torch.arange(L)[None]
        return ((tokens == 3) & (pos < lens[:, None])).float().sum(1)

    def greedy_reward():
        tokens, _ = scst.rollout(core, feats, length, sample=False, max_len=L)
        return reward(tokens, length).mean().item()

    before = greedy_reward()
    for _ in range(40):
        tokens, steps = scst.rollout(core, rep_feats, rep_len, sample=True, max_len=L)
        greedy, _ = scst.rollout(core, feats, length, sample=False, max_len=L)
        adv = reward(tokens, rep_len) - reward(greedy, length).repeat_interleave(n)
        opt.zero_grad()
        scst.policy_gradient_backward(model, core, rep_feats, rep_len, steps, adv, scaler, replay_steps=2)
        opt.step()
    after = greedy_reward()
    assert after > before + 1.0, (before, after)


def test_cider_d():
    corpus = [['a man riding a horse', 'a person on a horse in a field'],
              ['two cats sleeping on a couch', 'cats asleep on the sofa'],
              ['a plate of food with broccoli', 'broccoli and rice on a plate']]
    cider = scst.CiderD(corpus)
    refs = corpus[0]
    good, other, empty = cider.score(['A man riding a horse.', 'two cats on a couch', ''], refs)
    assert good > other > 0 or (good > 0 and other == 0)
    assert empty == 0
    assert np.isclose(cider.score(['a man riding a horse'], refs)[0], good)


def test_decode_captions_stops_at_eos():
    class Tok:
        eos_token_id = 9

        def decode(self, ids):
            return ' '.join(f'w{i}' for i in ids)

    tokens = torch.tensor([[1, 2, 9, 4], [5, 6, 7, 8]])
    assert scst.decode_captions(tokens, torch.tensor([4, 2]), Tok()) == ['w1 w2', 'w5 w6']
