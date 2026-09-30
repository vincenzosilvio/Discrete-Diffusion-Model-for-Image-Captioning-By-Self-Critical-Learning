"""Self-critical sequence training (SCST) for the discrete diffusion captioner.

The policy is the sampler the model is evaluated with (generate2_adpt_if in misc.py):
predict a length L, start from L mask tokens, and at each of L steps commit one token
in one masked slot. Committed tokens never change, so the log-probability of a caption
is an exact sum of per-step terms:

    log pi(c | I) = log p(L | I) + sum_k [ log p(slot_k | x_k) + log p(w_k | x_k, slot_k) ]

The slot term is only present when slots are sampled; with the default argmax slot the
slot is a deterministic function of the state and contributes no gradient. The length
term is only present when the length is sampled.

Rollouts run without grad and record every (x_k, t_k, slot_k, w_k). The loss then
replays a subset of those steps with grad, one chunk at a time, so memory is one forward
pass per chunk instead of a graph unrolled through all steps.
"""
import contextlib
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- reward

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_caption(s):
    """Lowercase and drop punctuation, so GPT-2 decodes and COCO references split alike."""
    return " ".join(_WORD_RE.findall(s.lower()))


def _ngram_counts(words, n):
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            counts[tuple(words[i:i + k])] += 1
    return counts


class CiderD:
    """CIDEr-D (Vedantam et al., 2015), as in the coco-caption / ruotianluo scorer.

    Document frequencies come from a fixed reference corpus (one list of reference
    strings per image), so scores do not depend on what else is in the batch.
    """

    def __init__(self, corpus_refs, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.doc_freq = defaultdict(float)
        num_images = 0
        for refs in corpus_refs:
            seen = set()
            for r in refs:
                seen.update(_ngram_counts(normalize_caption(r).split(), n))
            for g in seen:
                self.doc_freq[g] += 1.0
            num_images += 1
        self.ref_len = np.log(float(num_images))

    def _vec(self, caption):
        counts = _ngram_counts(normalize_caption(caption).split(), self.n)
        vec = [dict() for _ in range(self.n)]
        norm = np.zeros(self.n)
        length = 0
        for g, tf in counts.items():
            k = len(g) - 1
            df = np.log(max(1.0, self.doc_freq.get(g, 0.0)))
            vec[k][g] = float(tf) * (self.ref_len - df)
            norm[k] += vec[k][g] ** 2
            if k == 1:
                length += tf
        return vec, np.sqrt(norm), length

    def _sim(self, hyp, ref):
        (vh, nh, lh), (vr, nr, lr) = hyp, ref
        val = np.zeros(self.n)
        for k in range(self.n):
            for g, c in vh[k].items():
                r = vr[k].get(g)
                if r is not None:
                    val[k] += min(c, r) * r
            if nh[k] != 0 and nr[k] != 0:
                val[k] /= nh[k] * nr[k]
            val[k] *= np.e ** (-(float(lh - lr) ** 2) / (2 * self.sigma ** 2))
        return val

    def score(self, captions, refs):
        """CIDEr-D of each caption in `captions` against the same list of references."""
        ref_vecs = [self._vec(r) for r in refs]
        out = np.zeros(len(captions))
        for i, c in enumerate(captions):
            hyp = self._vec(c)
            s = sum(self._sim(hyp, r) for r in ref_vecs)
            out[i] = np.mean(s) / len(ref_vecs) * 10.0
        return out


def decode_captions(tokens, length, tokenizer):
    """Token ids -> text, cut at the predicted length and at the first <|endoftext|>."""
    eos = tokenizer.eos_token_id
    captions = []
    for row, l in zip(tokens.tolist(), length.tolist()):
        row = row[:l]
        if eos in row:
            row = row[:row.index(eos)]
        captions.append(tokenizer.decode(row).strip())
    return captions


# ----------------------------------------------------------------------------- policy

def build_attention_mask(state, mask_id):
    """Masked slots cannot be attended to except by themselves (as in generate2_adpt_if)."""
    L = state.size(1)
    attn = (state == mask_id).float().mul(-10000.0).unsqueeze(1).repeat(1, L, 1)
    idx = torch.arange(L, device=state.device)
    attn[:, idx, idx] = 0.0
    return attn


def step_timestep(i, length, time_step):
    """Diffusion timestep fed to the model at unmasking step i (as in generate2_adpt_if)."""
    if i == 0:
        return torch.full_like(length, time_step - 1)
    return (((length - i) * time_step) // length).clamp(min=0)


def policy_log_probs(fwd, mask_id, state, feats, t, guidance_scale=1.0):
    """log p(x0 | x_t) over the vocabulary at every slot, (N, L, V), plus the length logits.

    `fwd` is ClipCaptionModel (or its DDP wrapper) called with precomputed image features.
    With guidance != 1 the distribution is the classifier-free guided one used at test
    time; the conditional and image-free halves go through one forward pass so DDP sees
    a single forward per backward.
    """
    tokens = state.masked_fill(state == mask_id, 0)  # mask slots are replaced by bos_embedding
    attn = build_attention_mask(state, mask_id)
    if guidance_scale == 1.0:
        out, len_logits = fwd(tokens, state, None, attn, t, image_feats=feats)
        return F.log_softmax(out.logits.float(), dim=-1), len_logits
    n = state.size(0)

    def two(x):
        return torch.cat([x, x], 0)

    image_free = torch.arange(2 * n, device=state.device) >= n
    out, len_logits = fwd(two(tokens), two(state), None, two(attn), two(t),
                          image_feats=tuple(two(f) for f in feats), image_free=image_free)
    logp = F.log_softmax(out.logits.float(), dim=-1)
    guided = guidance_scale * (logp[:n] - logp[n:]) + logp[n:]
    return F.log_softmax(guided, dim=-1), len_logits[:n]


def _slot_scores(logp, state, mask_id, length):
    """Confidence of each slot (log-prob of its best token); -inf where it cannot be picked."""
    pos = torch.arange(state.size(1), device=state.device)
    open_slots = (state == mask_id) & (pos[None, :] < length[:, None])
    return logp.max(-1).values.masked_fill(~open_slots, float('-inf'))


def _action_log_prob(logp, state, mask_id, length, slot, token, sample_slot):
    rows = torch.arange(state.size(0), device=state.device)
    lp = logp[rows, slot, token]
    if sample_slot:
        lp = lp + F.log_softmax(_slot_scores(logp, state, mask_id, length), dim=-1)[rows, slot]
    return lp


@torch.no_grad()
def rollout(core, feats, length, sample, guidance_scale=1.0, sample_slot=False, max_len=20):
    """Run the unmasking sampler. Returns the final tokens (N, max_len) and, when
    `sample` is set, the list of recorded steps for `policy_gradient_backward`.

    With sample=False this is the greedy decoder: argmax slot, argmax token.
    """
    mask_id = core.num_classes - 1
    n, device = length.size(0), length.device
    assert int(length.max()) <= min(core.time_step, max_len), 'one token per step needs length <= time_step'
    rows = torch.arange(n, device=device)
    state = torch.full((n, max_len), mask_id, dtype=torch.long, device=device)
    steps = []
    for i in range(int(length.max())):
        t = step_timestep(i, length, core.time_step)
        logp, _ = policy_log_probs(core, mask_id, state, feats, t, guidance_scale)
        active = length > i  # finished captions have no open slot left
        scores = _slot_scores(logp, state, mask_id, length).masked_fill(~active[:, None], 0.0)
        if sample and sample_slot:
            slot = torch.multinomial(F.softmax(scores, dim=-1), 1).squeeze(1)
        else:
            slot = scores.argmax(-1)
        token_logp = logp[rows, slot]
        if sample:
            token = torch.multinomial(token_logp.exp(), 1).squeeze(1)
            lp = _action_log_prob(logp, state, mask_id, length, slot, token, sample_slot)
            steps.append({'state': state.clone(), 't': t, 'slot': slot, 'token': token,
                          'logp': torch.where(active, lp, torch.zeros_like(lp))})
        else:
            token = token_logp.argmax(-1)
        state[rows[active], slot[active]] = token[active]
    return state, steps


def _replay_plan(length, replay_steps, generator=None):
    """Which (step, caption) pairs to replay with grad, and the weight of each.

    replay_steps <= 0 replays every step. Otherwise each caption replays that many
    steps drawn uniformly without replacement, weighted by K / m so the sum over the
    replayed steps is an unbiased estimate of the full sum.
    """
    step_idx, cap_idx, weight, first = [], [], [], []
    for j, k in enumerate(length.tolist()):
        if replay_steps <= 0 or replay_steps >= k:
            chosen = list(range(k))
        else:
            chosen = torch.randperm(k, generator=generator)[:replay_steps].tolist()
        step_idx += chosen
        cap_idx += [j] * len(chosen)
        weight += [k / len(chosen)] * len(chosen)
        first += [True] + [False] * (len(chosen) - 1)
    dev = length.device
    return (torch.tensor(step_idx, device=dev), torch.tensor(cap_idx, device=dev),
            torch.tensor(weight, device=dev, dtype=torch.float), torch.tensor(first, device=dev))


def policy_gradient_backward(model, core, feats, length, steps, advantage, scaler,
                             guidance_scale=1.0, sample_slot=False, sample_length=False,
                             replay_steps=4, chunk_size=None, amp_enabled=False, generator=None):
    """Backpropagate -A * log pi(caption), normalised by the number of tokens.

    Replays the recorded steps with grad in chunks, calling backward once per chunk.
    Every chunk but the last runs under DDP's no_sync(), so gradients are all-reduced
    once. Returns (loss value, mean |replayed - rollout| log-prob of the replayed steps);
    the second should be ~0 and is a check that replay reproduces the rollout policy.
    """
    mask_id = core.num_classes - 1
    n = length.size(0)
    chunk_size = chunk_size or n
    step_idx, cap_idx, weight, first = _replay_plan(length, replay_steps, generator)
    stacked = {k: torch.stack([s[k] for s in steps]) for k in ('state', 't', 'slot', 'token', 'logp')}
    denom = length.sum().float()
    total_loss, gap_sum = 0.0, 0.0
    num_rows = step_idx.size(0)
    starts = list(range(0, num_rows, chunk_size))
    for c, start in enumerate(starts):
        sl = slice(start, start + chunk_size)
        si, ci = step_idx[sl], cap_idx[sl]
        state = stacked['state'][si, ci]
        chunk_feats = tuple(f[ci] for f in feats)
        last = c == len(starts) - 1
        sync = contextlib.nullcontext() if last or not hasattr(model, 'no_sync') else model.no_sync()
        with sync:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logp, len_logits = policy_log_probs(model, mask_id, state, chunk_feats,
                                                    stacked['t'][si, ci], guidance_scale)
            lp = _action_log_prob(logp, state, mask_id, length[ci], stacked['slot'][si, ci],
                                  stacked['token'][si, ci], sample_slot)
            gap_sum += (lp.detach() - stacked['logp'][si, ci]).abs().sum().item()
            term = weight[sl] * lp
            if sample_length:
                rows = torch.arange(state.size(0), device=state.device)
                len_lp = F.log_softmax(len_logits.float(), dim=-1)[rows, length[ci] - 1]
                term = term + first[sl].float() * len_lp
            loss = -(advantage[ci] * term).sum() / denom
            # keep every trainable parameter in the graph so DDP's reducer finishes
            loss = loss + 0.0 * len_logits.float().sum() + 0.0 * core.pad_embedding.sum().float()
            scaler.scale(loss).backward()
        total_loss += loss.item()
    return total_loss, gap_sum / max(num_rows, 1)
