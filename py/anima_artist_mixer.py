"""LZ Anima Artist Mixer - Anima-Artist-Mixer derived nodes for LZNodes.

MIT License

Copyright (c) 2026 An1X3R and �汐浮尘

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This file is derived from Anima-Artist-Mixer (https://github.com/An1X3R/Anima-Artist-Mixer).
Modifications by Liz for ComfyUI-LZNodes:
- LZ-prefixed class names to avoid collisions
- Added STRING output (positive_text) to LZAnimaArtistCrossAttn
- Added LZAnimaArtistNode combined node (Pack + CrossAttn + STRING output)
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_BASE_PRESERVE = "base_preserve"

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_LOWRANK_AVG = "lowrank_avg"

MAX_ARTISTS = 32

_STATIC_CAPTURE_K_DEFAULT = 6
_STATIC_CAPTURE_K_MAX = 12

_ANCHOR_SEED = 42
_ANCHOR_SEEDS_POOL = [42, 100, 200, 300]
_ANCHOR_SEEDS_MAX = 4
_ANCHOR_LAYER_THRESHOLD_DISABLED = -1


def _extract(conditioning):
    if conditioning is None:
        return None, None, None
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        return None, None, None
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        return None, None, None
    raw = first[0] if torch.is_tensor(first[0]) else None
    extra = first[1] if len(first) > 1 and isinstance(first[1], dict) else {}
    return raw, extra.get("t5xxl_ids"), extra.get("t5xxl_weights")


def _split_artist_chain(chain):
    if not chain:
        return []
    s = str(chain).replace("\uff0c", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _parse_artist_weights(parts):
    names = []
    weights = []
    has_explicit = False
    for raw in parts:
        s = str(raw).strip()
        if not s:
            continue
        weight = 1.0
        explicit = False
        if "::" in s:
            head = s
            if head.startswith("::"):
                head = head[2:]
            if "::" in head:
                name_part, _, w_part = head.rpartition("::")
                w_part = w_part.strip()
                try:
                    w_val = float(w_part)
                    weight = max(0.0, min(4.0, w_val))
                    explicit = True
                    s = name_part.strip()
                except ValueError:
                    pass
        if not s:
            continue
        names.append(s)
        weights.append(weight)
        if explicit:
            has_explicit = True
    return names, weights, has_explicit


def _parse_layer_filter(text, num_blocks):
    if not text:
        return None
    s = str(text).replace("\uff0c", ",").replace(" ", "")
    if not s:
        return None
    result = set()
    for part in s.split(","):
        if not part:
            continue
        if "-" in part[1:]:
            dash_idx = part.index("-", 1)
            try:
                lo = int(part[:dash_idx])
                hi = int(part[dash_idx + 1:])
            except ValueError:
                continue
            if lo < 0:
                lo += num_blocks
            if hi < 0:
                hi += num_blocks
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(num_blocks - 1, hi)
            if lo <= hi:
                result.update(range(lo, hi + 1))
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            if v < 0:
                v += num_blocks
            if 0 <= v < num_blocks:
                result.add(v)
    return sorted(result) if result else None


def _normalize_weights(weights):
    total = sum(abs(w) for w in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def _project_perpendicular(delta, base):
    base_norm_sq = (base * base).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (delta * base).sum(dim=-1, keepdim=True) / base_norm_sq
    return delta - proj_coef * base


def _unwrap_cross_attn(ca):
    while isinstance(ca, _CrossAttnWrapper):
        ca = ca.original
    return ca


def _validate(diffusion_model):
    if not hasattr(diffusion_model, "blocks"):
        return False, 0, 0, f"{type(diffusion_model).__name__} has no .blocks"
    blocks = diffusion_model.blocks
    if len(blocks) == 0:
        return False, 0, 0, ".blocks is empty"
    b0 = blocks[0]
    if not hasattr(b0, "cross_attn"):
        return False, 0, 0, "blocks[0] has no cross_attn"
    ca = _unwrap_cross_attn(b0.cross_attn)
    if not hasattr(ca, "context_dim"):
        return False, 0, 0, "cross_attn has no context_dim"
    return True, len(blocks), int(ca.context_dim), "ok"


def _cleanup_residual_wrappers(dm):
    if not hasattr(dm, "blocks"):
        return 0
    cleaned = 0
    for i in range(len(dm.blocks)):
        blk = dm.blocks[i]
        if not hasattr(blk, "cross_attn"):
            continue
        original = _unwrap_cross_attn(blk.cross_attn)
        if blk.cross_attn is not original:
            blk.cross_attn = original
            cleaned += 1
    return cleaned


def _preprocess_one(dm, raw, ids, weights, target_device, target_dtype):
    if ids is None:
        artist = raw.to(device=target_device, dtype=target_dtype)
        if artist.dim() == 2:
            artist = artist.unsqueeze(0)
        return artist
    raw_b = raw if raw.dim() == 3 else raw.unsqueeze(0)
    ids_b = ids if ids.dim() >= 2 else ids.unsqueeze(0)
    weights_b = None
    if weights is not None:
        if weights.dim() == 1:
            weights_b = weights.unsqueeze(0).unsqueeze(-1)
        elif weights.dim() == 2:
            weights_b = weights.unsqueeze(-1)
        else:
            weights_b = weights
    raw_b = raw_b.to(device=target_device, dtype=target_dtype)
    ids_b = ids_b.to(device=target_device)
    if weights_b is not None:
        weights_b = weights_b.to(device=target_device, dtype=target_dtype)
    with torch.inference_mode():
        return dm.preprocess_text_embeds(raw_b, ids_b, t5xxl_weights=weights_b)


def _build_artists(state, ref_context):
    if state.get("individuals") is not None:
        return state["individuals"], state["real_lens"]
    dm = state["dm_ref"]
    individuals, real_lens = [], []
    for raw, ids, w_t in zip(state["raws"], state["ids_list"], state["w_list"]):
        artist = _preprocess_one(dm, raw, ids, w_t, ref_context.device, ref_context.dtype)
        individuals.append(artist)
        real_lens.append(int(ids.shape[-1]) if ids is not None else artist.shape[1])
    state["individuals"] = individuals
    state["real_lens"] = real_lens
    return individuals, real_lens


def _combine_concat(individuals, weights):
    parts = [a * float(w) for a, w in zip(individuals, weights)]
    return torch.cat(parts, dim=1)


def _broadcast_batch(t, batch_size):
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, -1, -1)
    if batch_size % t.shape[0] == 0:
        return t.repeat(batch_size // t.shape[0], 1, 1)
    return t[:1].expand(batch_size, -1, -1)


def _resolve_mask(cou, batch_size, apply_to_uncond, state):
    if cou is None or len(cou) != batch_size:
        if not state.get("_warned", False):
            logger.warning(
                "[AnimaCrossAttn] cond_or_uncond not available (got=%s, batch=%d), "
                "falling back to inject all rows.", cou, batch_size,
            )
            state["_warned"] = True
        return [True] * batch_size
    if apply_to_uncond:
        return [True] * batch_size
    return [c == 0 for c in cou]


def _in_sigma_range(state):
    rng = state.get("sigma_range")
    if rng is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    lo, hi = rng
    return lo <= cur <= hi


class _CrossAttnWrapper(nn.Module):
    def __init__(self, original, shared_state, layer_idx):
        super().__init__()
        self.original = original
        self._st = shared_state
        self._idx = layer_idx
        self._disabled = False

    def _maybe_reset_ema(self):
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev = st.get("_ema_last_sigma")
        if prev is None or cur > prev + 1e-3:
            st["_ema_cache"] = {}
        st["_ema_last_sigma"] = cur

    def _apply_ema(self, artist_total, fusion_mode):
        if self._st.get("artist_static_capture", False):
            return artist_total
        ema_alpha = float(self._st.get("artist_ema_alpha", 0.0))
        ema_compatible = fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        if ema_alpha <= 0.0 or not ema_compatible:
            return artist_total
        self._maybe_reset_ema()
        cache = self._st.setdefault("_ema_cache", {})
        prev = cache.get(self._idx)
        if prev is not None and prev.shape == artist_total.shape:
            artist_total = ema_alpha * prev + (1.0 - ema_alpha) * artist_total
        cache[self._idx] = artist_total.detach()
        return artist_total

    def _maybe_reset_static(self):
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev_max = st.get("_static_max_sigma")
        if prev_max is None or cur > prev_max + 1e-3:
            st["_static_cache"] = {}
            st["_static_max_sigma"] = cur

    def _get_artist_outputs_with_cache(self, x, context, rope_emb, t_opts,
                                        individuals, fusion_mode):
        st = self._st
        if not st.get("artist_static_capture", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        self._maybe_reset_static()
        cache = st.setdefault("_static_cache", {})
        n = len(individuals)
        fp = (tuple(x.shape), n)

        cur_sigma = st.get("current_sigma")
        sigma_key = round(float(cur_sigma), 4) if cur_sigma is not None else None

        entry = cache.get(self._idx)
        if entry is None or entry.get("_fp") != fp:
            entry = {
                "_fp": fp,
                "seen_sigmas": set(),
                "accumulator": None,
                "count": 0,
                "frozen": False,
                "frozen_outputs": None,
            }
            cache[self._idx] = entry

        if entry["frozen"]:
            return entry["frozen_outputs"]

        if sigma_key is not None and sigma_key in entry["seen_sigmas"]:
            if entry["accumulator"] is not None and entry["count"] > 0:
                inv = 1.0 / entry["count"]
                out_dtype = context.dtype
                return [(a * inv).to(out_dtype) for a in entry["accumulator"]]
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        outs = self._collect_artist_outputs(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )
        out_dtype = outs[0].dtype
        if entry["accumulator"] is None:
            entry["accumulator"] = [o.detach().to(torch.float32) for o in outs]
        else:
            for i, o in enumerate(outs):
                entry["accumulator"][i] = entry["accumulator"][i] + o.detach().to(torch.float32)
        entry["count"] += 1
        if sigma_key is not None:
            entry["seen_sigmas"].add(sigma_key)

        capture_k = int(self._st.get("static_capture_k", _STATIC_CAPTURE_K_DEFAULT))
        if entry["count"] >= capture_k:
            inv = 1.0 / entry["count"]
            entry["frozen_outputs"] = [(a * inv).to(out_dtype) for a in entry["accumulator"]]
            entry["frozen"] = True
            entry["accumulator"] = None
            entry["seen_sigmas"] = None
            return entry["frozen_outputs"]

        inv = 1.0 / entry["count"]
        return [(a * inv).to(out_dtype) for a in entry["accumulator"]]

    def _apply_fusion(self, base_out, artist_total, mask, fusion_mode, strength):
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = artist_total - base_out
            delta_perp = _project_perpendicular(delta, base_out)
            out = base_out.clone()
            for i, hit in enumerate(mask):
                if hit:
                    out[i] = base_out[i] + strength * delta_perp[i]
            return out

        out = base_out.clone()
        for i, hit in enumerate(mask):
            if hit:
                out[i] = base_out[i] * (1.0 - strength) + artist_total[i] * strength
        return out

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        st = self._st

        if st.get("_in_anchor_run", False):
            cache = st.setdefault("_anchor_cache", {})
            cache[self._idx] = x.detach().clone()
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not st.get("enabled", False) or context is None:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if self._disabled:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not _in_sigma_range(st):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        try:
            return self._dispatch(x, context, rope_emb, transformer_options)
        except Exception as e:
            logger.exception(
                "[AnimaCrossAttn] L%d injection path exception, "
                "falling back to original cross_attn: %s",
                self._idx, e,
            )
            self._disabled = True
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        individuals, _ = _build_artists(st, context)
        combine_mode = st["combine_mode"]
        fusion_mode = st["fusion_mode"]
        strength = float(st["strength"])
        weights = st["user_weights"]

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        bsz = context.shape[0]
        mask = _resolve_mask(cou, bsz, st["apply_to_uncond"], st)

        if not any(mask):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength,
            )

        if combine_mode == COMBINE_OUTPUT_AVG or combine_mode == COMBINE_LOWRANK_AVG:
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength,
            )

        combined = _combine_concat(individuals, weights)
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strength,
        )

    def _fwd_output_avg(self, x, context, rope_emb, t_opts,
                        individuals, weights, mask, fusion_mode, strength):
        bsz = context.shape[0]

        if self._st.get("normalize_weights", True):
            ws = _normalize_weights(weights)
        else:
            ws = list(weights)
        n = len(individuals)
        static_capture = self._st.get("artist_static_capture", False)
        force_collect = static_capture and fusion_mode != FUSION_CONCAT_WITH_BASE

        artist_total = None
        if force_collect:
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
            for out_i, w in zip(outs, ws):
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w
        elif n >= 2 and not self._st.get("_disable_batched", False):
            try:
                q_x = self._get_anchor_q_x(x)
                artist_total = self._batched_artists_forward(
                    q_x, context, rope_emb, t_opts, individuals, ws, fusion_mode
                )
            except Exception as e:
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched output_avg failed, "
                        "falling back to serial: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
                artist_total = None
        if artist_total is None:
            q_x = self._get_anchor_q_x(x)
            for artist_i, w in zip(individuals, ws):
                artist_b = _broadcast_batch(artist_i, bsz).to(
                    device=context.device, dtype=context.dtype)
                kv = torch.cat([context, artist_b], dim=1) \
                    if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
                out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w

        artist_total = self._apply_ema(artist_total, fusion_mode)

        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _get_anchor_q_x(self, x):
        st = self._st
        if not st.get("artist_anchor_q", False):
            return x
        if st.get("_anchor_failed", False):
            return x

        threshold = int(st.get("anchor_deep_layer_threshold", _ANCHOR_LAYER_THRESHOLD_DISABLED))
        if threshold >= 0 and self._idx >= threshold:
            return x

        cache = st.get("_anchor_cache", {})
        anchor_x = cache.get(self._idx)
        if anchor_x is None:
            return x
        if anchor_x.shape != x.shape:
            if anchor_x.shape[1:] == x.shape[1:]:
                ax_bsz = anchor_x.shape[0]
                bsz = x.shape[0]
                if bsz % ax_bsz == 0:
                    anchor_x = anchor_x.repeat(bsz // ax_bsz, *([1] * (anchor_x.dim() - 1)))
                elif ax_bsz % bsz == 0:
                    anchor_x = anchor_x[:bsz]
                else:
                    return x
            else:
                return x
        anchor_x = anchor_x.to(device=x.device, dtype=x.dtype)

        blend = float(st.get("anchor_user_blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        if blend > 0.0:
            return blend * x + (1.0 - blend) * anchor_x
        return anchor_x

    def _collect_artist_outputs(self, x, context, rope_emb, t_opts,
                                individuals, fusion_mode):
        bsz = context.shape[0]
        n = len(individuals)
        q_x = self._get_anchor_q_x(x)
        if n >= 2 and not self._st.get("_disable_batched", False):
            try:
                return self._batched_artists_outputs_only(
                    q_x, context, rope_emb, t_opts, individuals, fusion_mode
                )
            except Exception as e:
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched outputs failed, "
                        "falling back to serial: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
        outs = []
        for artist_i in individuals:
            artist_b = _broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            kv = torch.cat([context, artist_b], dim=1) \
                if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
            out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
            outs.append(out_i)
        return outs

    def _batched_artists_outputs_only(self, x, context, rope_emb, t_opts,
                                       individuals, fusion_mode):
        bsz = context.shape[0]
        n = len(individuals)
        kv_list = []
        for artist_i in individuals:
            artist_b = _broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(torch.cat([context, artist_b], dim=1))
            else:
                kv_list.append(artist_b)
        kv_lens = {kv.shape[1] for kv in kv_list}
        if len(kv_lens) > 1:
            raise ValueError(f"K/V lengths mismatch {kv_lens}, cannot batch")
        x_rep = x.repeat(n, *([1] * (x.dim() - 1)))
        kv_stacked = torch.cat(kv_list, dim=0)
        rope_rep = rope_emb
        if rope_emb is not None and torch.is_tensor(rope_emb):
            if rope_emb.dim() > 0 and rope_emb.shape[0] == bsz:
                rope_rep = rope_emb.repeat(n, *([1] * (rope_emb.dim() - 1)))
        new_opts = dict(t_opts) if isinstance(t_opts, dict) else {}
        cou = new_opts.get("cond_or_uncond")
        if cou is not None:
            new_opts["cond_or_uncond"] = list(cou) * n
        out = self.original(x_rep, kv_stacked, rope_emb=rope_rep,
                            transformer_options=new_opts)
        out = out.view(n, bsz, *out.shape[1:])
        return [out[i] for i in range(n)]

    def _fwd_lowrank_avg(self, x, context, rope_emb, t_opts,
                         individuals, weights, mask, fusion_mode, strength):
        if self._st.get("normalize_weights", True):
            ws = _normalize_weights(weights)
        else:
            ws = list(weights)
        n = len(individuals)
        k = int(self._st.get("lowrank_k", 1))
        k = max(1, min(k, n))

        artist_outs = self._get_artist_outputs_with_cache(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )

        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        out_dtype = base_out.dtype

        A = torch.stack(artist_outs, dim=0).to(torch.float32)
        base_f32 = base_out.to(torch.float32).unsqueeze(0)
        delta = A - base_f32

        orig_shape = delta.shape
        D_mat = delta.reshape(n, -1)

        if k < n:
            try:
                U, S, V = torch.svd_lowrank(D_mat, q=k, niter=2)
                D_lowrank = U @ torch.diag(S) @ V.transpose(-1, -2)
            except Exception as e:
                if not self._st.get("_warned_svd", False):
                    logger.warning(
                        "[AnimaCrossAttn] L%d SVD failed, "
                        "falling back to output_avg this step: %s",
                        self._idx, e,
                    )
                    self._st["_warned_svd"] = True
                D_lowrank = D_mat
        else:
            D_lowrank = D_mat

        w_t = torch.tensor(ws, device=D_lowrank.device, dtype=D_lowrank.dtype).view(n, 1)
        delta_avg = (D_lowrank * w_t).sum(dim=0)
        delta_avg = delta_avg.reshape(orig_shape[1:]).to(out_dtype)

        artist_total = base_out + delta_avg

        artist_total = self._apply_ema(artist_total, fusion_mode)

        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _batched_artists_forward(self, x, context, rope_emb, t_opts,
                                 individuals, weights, fusion_mode):
        bsz = context.shape[0]
        n = len(individuals)
        kv_list = []
        for artist_i in individuals:
            artist_b = _broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(torch.cat([context, artist_b], dim=1))
            else:
                kv_list.append(artist_b)
        kv_lens = {kv.shape[1] for kv in kv_list}
        if len(kv_lens) > 1:
            raise ValueError(f"K/V lengths mismatch {kv_lens}, cannot batch")
        x_rep = x.repeat(n, *([1] * (x.dim() - 1)))
        kv_stacked = torch.cat(kv_list, dim=0)
        rope_rep = rope_emb
        if rope_emb is not None and torch.is_tensor(rope_emb):
            if rope_emb.dim() > 0 and rope_emb.shape[0] == bsz:
                rope_rep = rope_emb.repeat(n, *([1] * (rope_emb.dim() - 1)))
        new_opts = dict(t_opts) if isinstance(t_opts, dict) else {}
        cou = new_opts.get("cond_or_uncond")
        if cou is not None:
            new_opts["cond_or_uncond"] = list(cou) * n
        out = self.original(x_rep, kv_stacked, rope_emb=rope_rep,
                            transformer_options=new_opts)
        out = out.view(n, bsz, *out.shape[1:])
        w_t = torch.tensor(weights, device=out.device, dtype=out.dtype).view(
            n, *([1] * (out.dim() - 1))
        )
        return (out * w_t).sum(dim=0)

    def _fwd_with_combined(self, x, context, rope_emb, t_opts,
                          combined, mask, fusion_mode, strength):
        bsz = context.shape[0]
        artist_b = _broadcast_batch(combined, bsz).to(
            device=context.device, dtype=context.dtype)

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            q_x = self._get_anchor_q_x(x)
            static_capture = self._st.get("artist_static_capture", False)
            if static_capture:
                self._maybe_reset_static()
                cache = self._st.setdefault("_static_cache", {})
                cached = cache.get(self._idx)
                fp = (tuple(x.shape), -1)
                if cached is not None and cached.get("_fp") == fp:
                    artist_out = cached["outputs"][0]
                else:
                    artist_out = self.original(q_x, artist_b, rope_emb=rope_emb, transformer_options=t_opts)
                    cache[self._idx] = {"outputs": [artist_out.detach()], "_fp": fp}
            else:
                artist_out = self.original(q_x, artist_b, rope_emb=rope_emb, transformer_options=t_opts)
            artist_out = self._apply_ema(artist_out, fusion_mode)

            if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
                return artist_out
            return self._apply_fusion(base_out, artist_out, mask, fusion_mode, strength)

        artist_len = artist_b.shape[1]
        extension = torch.zeros(bsz, artist_len, context.shape[-1],
                                device=context.device, dtype=context.dtype)
        for i, hit in enumerate(mask):
            if hit:
                extension[i] = artist_b[i]
        merged = torch.cat([context, extension], dim=1)
        return self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)


def _make_sigma_capture(state, prev_wrapper):
    def wrapper(apply_model, options):
        ts = options.get("timestep")
        cur_sigma = None
        if ts is not None:
            try:
                cur_sigma = float(ts.flatten()[0].item())
                state["current_sigma"] = cur_sigma
            except Exception:
                pass

        if state.get("artist_anchor_q", False) and not state.get("_anchor_failed", False):
            user_x = options.get("input")
            user_ts = options.get("timestep")
            c_dict = options.get("c", {}) or {}
            if user_x is not None and user_ts is not None and c_dict:
                _maybe_run_anchor(state, user_x, user_ts, c_dict)

        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper


def _maybe_run_anchor(state, user_x, user_timestep, c_dict):
    base_context = c_dict.get("context")
    if base_context is None:
        return

    transformer_options = c_dict.get("transformer_options", {}) or {}
    if base_context.dim() >= 2 and base_context.shape[0] > 1:
        cou = transformer_options.get("cond_or_uncond")
        if cou is not None and 0 in cou:
            cond_idx = cou.index(0)
            base_context = base_context[cond_idx:cond_idx + 1]
        else:
            base_context = base_context[:1]

    cache_key = state.get("_anchor_cache_key")
    new_key = (tuple(user_x.shape), id(c_dict.get("context")))
    if cache_key == new_key and state.get("_anchor_cache"):
        return

    dm = state["dm_ref"]

    state["_anchor_cache"] = {}
    state["_in_anchor_run"] = True

    bsz = user_x.shape[0]
    if base_context.shape[0] != bsz:
        if base_context.shape[0] == 1:
            ctx_for_anchor = base_context.expand(bsz, -1, -1)
        else:
            ctx_for_anchor = base_context[:1].expand(bsz, -1, -1)
    else:
        ctx_for_anchor = base_context
    ctx_for_anchor = ctx_for_anchor.contiguous().to(device=user_x.device, dtype=user_x.dtype)

    anchor_kwargs = {}
    for key in ("t5xxl_ids", "t5xxl_weights"):
        v = c_dict.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if v.shape[0] != bsz:
            if v.shape[0] == 1:
                v = v.expand(bsz, *v.shape[1:])
            else:
                v = v[:1].expand(bsz, *v.shape[1:])
        anchor_kwargs[key] = v.contiguous()

    safe_opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    safe_opts.pop("cond_or_uncond", None)
    safe_opts.pop("patches", None)
    anchor_kwargs["transformer_options"] = safe_opts

    try:
        with torch.no_grad():
            t5xxl_ids = anchor_kwargs.pop("t5xxl_ids", None)
            t5xxl_weights = anchor_kwargs.pop("t5xxl_weights", None)
            if t5xxl_ids is not None and hasattr(dm, "preprocess_text_embeds"):
                processed_ctx = dm.preprocess_text_embeds(
                    ctx_for_anchor, t5xxl_ids, t5xxl_weights=t5xxl_weights,
                )
            else:
                processed_ctx = ctx_for_anchor
            t_opts_for_anchor = anchor_kwargs.get("transformer_options", {})

            seeds_count = max(1, min(int(state.get("anchor_seeds_count", 1)), _ANCHOR_SEEDS_MAX))
            seeds = _ANCHOR_SEEDS_POOL[:seeds_count]

            accumulator = {}
            for seed in seeds:
                gen = torch.Generator(device=user_x.device)
                gen.manual_seed(seed)
                anchor_x_k = torch.randn(
                    user_x.shape, generator=gen,
                    device=user_x.device, dtype=user_x.dtype,
                )
                state["_anchor_cache"] = {}
                _ = dm._forward(
                    anchor_x_k, user_timestep, processed_ctx,
                    transformer_options=t_opts_for_anchor,
                )
                for layer_idx, hidden in state["_anchor_cache"].items():
                    if layer_idx not in accumulator:
                        accumulator[layer_idx] = hidden.to(torch.float32)
                    else:
                        accumulator[layer_idx] = accumulator[layer_idx] + hidden.to(torch.float32)

            inv = 1.0 / max(1, seeds_count)
            avg_dtype = user_x.dtype
            state["_anchor_cache"] = {
                idx: (acc * inv).to(avg_dtype) for idx, acc in accumulator.items()
            }
    except Exception as e:
        logger.warning(
            "[AnimaCrossAttn] anchor pre-run failed, "
            "falling back to v20 behavior: %s", e,
        )
        state["_anchor_cache"] = {}
        state["_anchor_failed"] = True
    finally:
        state["_in_anchor_run"] = False

    if state["_anchor_cache"]:
        state["_anchor_cache_key"] = new_key
        if not state.get("_warned_anchor_ok", False):
            logger.info(
                "[AnimaCrossAttn] anchor pre-run complete, "
                "captured %d layers' hidden states",
                len(state["_anchor_cache"]),
            )
            state["_warned_anchor_ok"] = True


class LZAnimaArtistPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Artist chain. Separate with commas or newlines.\n"
                        "Example: wlop, sakimichan, krenz\n"
                        "\n"
                        "Supports two weight syntaxes (can coexist):\n"
                        "  1) Bracket syntax (wlop:1.5) - affects CLIP encoding layer, nonlinear\n"
                        "  2) ::weight syntax ::wlop::1.5 - affects cross-attn injection layer, linear\n"
                        "\n"
                        "Default weight=1.0. Range [0.0, 4.0].\n"
                        "::weight and brackets can be stacked: ::(wlop:1.1)::0.8"
                    )
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Base prompt (optional). Recommend format: '<artist>\\n<base_prompt>'"
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PACK",)
    RETURN_NAMES = ("artist_pack",)
    FUNCTION = "pack"
    CATEGORY = "LZNodes/Anima"

    def pack(self, clip, artist_chain, base_prompt=""):
        parts = _split_artist_chain(artist_chain)
        names, parsed_weights, has_explicit = _parse_artist_weights(parts)
        base = (base_prompt or "").strip()

        try:
            base_tokens = clip.tokenize(base)
            base_conditioning = clip.encode_from_tokens_scheduled(base_tokens)
        except Exception as e:
            raise ValueError(
                f"[LZAnimaArtistPack] base_prompt encode failed (text={base!r}): {e}"
            )

        if not names:
            return ({
                "conditionings": [],
                "labels": [],
                "weights": [],
                "has_explicit_weights": False,
                "base_prompt": base,
                "base_conditioning": base_conditioning,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[LZAnimaArtistPack] Artist count %d exceeds max %d, truncating",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]

        conditionings = []
        for name in names:
            text = f"{name}\n{base}" if base else name
            try:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    f"[LZAnimaArtistPack] Encode failed (text={text!r}): {e}"
                )
            conditionings.append(cond)

        if has_explicit:
            logger.info(
                "[LZAnimaArtistPack] Detected %d artists with ::weight syntax, "
                "will use linear injection path",
                sum(1 for w in parsed_weights if w != 1.0),
            )

        return ({
            "conditionings": conditionings,
            "labels": names,
            "weights": parsed_weights,
            "has_explicit_weights": has_explicit,
            "base_prompt": base,
            "base_conditioning": base_conditioning,
        },)


class LZAnimaArtistOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 63, "step": 1,
                    "tooltip": "Starting block (inclusive). 0 = first layer"
                }),
                "end_block": ("INT", {
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "Ending block (inclusive). -1 = last layer"
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling progress start. 0.0 = sampling start"
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling progress end. 1.0 = sampling end"
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True: weights normalized to relative proportions. "
                        "False: weights used as independent strengths.\n"
                        "If ::weight syntax is used in artist_chain, "
                        "this switch is automatically disabled."
                    )
                }),
                "artist_ema_alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": (
                        "Cross-step EMA smoothing coefficient "
                        "(only effective for fusion=interpolate).\n"
                        "0.0: off (default)\n"
                        "0.3-0.5: light smoothing\n"
                        "0.5-0.8: medium-heavy smoothing\n"
                        ">0.8: strong smoothing\n"
                        "Auto-reset on new sampling (sigma rising)."
                    )
                }),
                "lowrank_k": ("INT", {
                    "default": 1, "min": 1, "max": MAX_ARTISTS, "step": 1,
                    "tooltip": (
                        "LoRA-style low-rank injection dimension "
                        "(only effective for combine_mode=lowrank_avg).\n"
                        "k=1: all artists along a single consensus direction\n"
                        "k=2-3: retain main style directions (recommended)\n"
                        "k>=N: equivalent to output_avg (no projection)"
                    )
                }),
                "artist_static_capture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "H' cross-step temporal averaging: "
                        "first K steps accumulate artist attention, "
                        "then freeze from step K onwards."
                    )
                }),
                "static_capture_k": ("INT", {
                    "default": _STATIC_CAPTURE_K_DEFAULT,
                    "min": 1, "max": _STATIC_CAPTURE_K_MAX, "step": 1,
                    "tooltip": "H' accumulation steps (only when artist_static_capture=True).\n"
                               "K=1: v18 single-point cache\n"
                               "K=6: recommended default"
                }),
                "artist_anchor_q": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use fixed-seed anchor hidden state for artist cross-attn Q, "
                        "fully decoupling from user seed. "
                        "Pre-runs one model forward on first use (~1s overhead)."
                    )
                }),
                "anchor_seeds_count": ("INT", {
                    "default": 1, "min": 1, "max": _ANCHOR_SEEDS_MAX, "step": 1,
                    "tooltip": (
                        "Number of fixed seeds for anchor pre-run "
                        "(only when anchor_q=True).\n"
                        "More seeds = less systematic bias, more overhead."
                    )
                }),
                "anchor_user_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Anchor/user x blend ratio (only when anchor_q=True).\n"
                        "Q = blend * user_x + (1-blend) * anchor_x\n"
                        "0.0: pure anchor; 1.0: pure user x."
                    )
                }),
                "anchor_deep_layer_threshold": ("INT", {
                    "default": _ANCHOR_LAYER_THRESHOLD_DISABLED,
                    "min": _ANCHOR_LAYER_THRESHOLD_DISABLED, "max": 64, "step": 1,
                    "tooltip": (
                        "Only use anchor in shallow layers (only when anchor_q=True).\n"
                        "-1: all layers use anchor (default)\n"
                        "N>=0: layer idx < N uses anchor, idx >= N uses user x"
                    )
                }),
            },
            "optional": {
                "layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Advanced layer selection (optional). "
                        "Comma-separated block indices, "
                        "supports ranges and negative indices.\n"
                        "Example: '0,3,5-10,-1'\n"
                        "Overrides start_block/end_block when set."
                    )
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "LZNodes/Anima"

    def build(self, start_block, end_block, start_percent, end_percent, normalize_weights,
              artist_ema_alpha=0.0, lowrank_k=1, artist_static_capture=False,
              static_capture_k=_STATIC_CAPTURE_K_DEFAULT, artist_anchor_q=False,
              anchor_seeds_count=1, anchor_user_blend=0.0,
              anchor_deep_layer_threshold=_ANCHOR_LAYER_THRESHOLD_DISABLED,
              layer_filter=""):
        return ({
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "normalize_weights": bool(normalize_weights),
            "artist_ema_alpha": float(artist_ema_alpha),
            "lowrank_k": int(lowrank_k),
            "artist_static_capture": bool(artist_static_capture),
            "static_capture_k": int(static_capture_k),
            "artist_anchor_q": bool(artist_anchor_q),
            "anchor_seeds_count": int(anchor_seeds_count),
            "anchor_user_blend": float(anchor_user_blend),
            "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
            "layer_filter": str(layer_filter or ""),
        },)


class LZAnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE],
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Artist injection strength.\n"
                        "0.0-1.0: interpolation lerp(base, artist, strength)\n"
                        "1.0-4.0: extrapolation base + strength * (artist - base)\n"
                        "Recommended 1.5-2.5 for extrapolation."
                    )
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "base_prompt", "positive_text")
    FUNCTION = "patch"
    CATEGORY = "LZNodes/Anima"

    def patch(self, model, artist_pack, combine_mode, fusion_mode,
              strength, enabled, apply_to_uncond, advanced_options=None):
        adv = advanced_options or {}
        sb = int(adv.get("start_block", 0))
        eb = int(adv.get("end_block", -1))
        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        artist_ema_alpha = float(adv.get("artist_ema_alpha", 0.0))
        lowrank_k = int(adv.get("lowrank_k", 1))
        artist_static_capture = bool(adv.get("artist_static_capture", False))
        static_capture_k = int(adv.get("static_capture_k", _STATIC_CAPTURE_K_DEFAULT))
        static_capture_k = max(1, min(static_capture_k, _STATIC_CAPTURE_K_MAX))
        artist_anchor_q = bool(adv.get("artist_anchor_q", False))
        anchor_seeds_count = int(adv.get("anchor_seeds_count", 1))
        anchor_seeds_count = max(1, min(anchor_seeds_count, _ANCHOR_SEEDS_MAX))
        anchor_user_blend = float(adv.get("anchor_user_blend", 0.0))
        anchor_user_blend = max(0.0, min(1.0, anchor_user_blend))
        anchor_deep_layer_threshold = int(
            adv.get("anchor_deep_layer_threshold", _ANCHOR_LAYER_THRESHOLD_DISABLED)
        )
        layer_filter_text = str(adv.get("layer_filter", "") or "")

        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)
        need_sigma_capture = (
            use_sigma_range or (artist_ema_alpha > 0.0)
            or artist_static_capture or artist_anchor_q
        )

        if artist_static_capture and artist_ema_alpha > 0.0:
            logger.info(
                "[LZAnimaCrossAttn] artist_static_capture=True, "
                "artist_ema_alpha=%.2f auto-ignored", artist_ema_alpha,
            )
        if artist_static_capture and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[LZAnimaCrossAttn] artist_static_capture=True but "
                "fusion=concat_with_base incompatible. Static auto-ignored."
            )
        if artist_anchor_q and artist_static_capture:
            logger.warning(
                "[LZAnimaCrossAttn] artist_anchor_q=True and "
                "artist_static_capture=True are mutually exclusive. "
                "Static disabled."
            )
            artist_static_capture = False
        if artist_anchor_q and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[LZAnimaCrossAttn] artist_anchor_q=True and "
                "fusion=concat_with_base incompatible. anchor_q disabled."
            )
            artist_anchor_q = False

        if fusion_mode == FUSION_BASE_PRESERVE and combine_mode == COMBINE_CONCAT:
            pass

        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[LZAnimaCrossAttn] artist_pack type error, "
                "please connect AnimaArtistPack node output"
            )

        conditionings = artist_pack.get("conditionings") or []
        labels = artist_pack.get("labels") or []

        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[LZAnimaCrossAttn] artist_pack missing base_conditioning field."
            )

        if not conditionings:
            return (model, base_cond_out, artist_pack.get("base_prompt", ""))

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = _extract(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[LZAnimaCrossAttn] artist[{label}] conditioning is empty."
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        n = len(raws)
        parsed_weights = artist_pack.get("weights")
        has_explicit_weights = bool(artist_pack.get("has_explicit_weights", False))
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == n:
            user_weights = [float(w) for w in parsed_weights]
        else:
            user_weights = [1.0] * n
            has_explicit_weights = False

        if fusion_mode == FUSION_BASE_PRESERVE and float(strength) < 0.3:
            logger.info(
                "[LZAnimaCrossAttn] fusion=base_preserve with strength=%.2f "
                "(<0.3) will be very weak.", float(strength),
            )

        if float(strength) > 1.0:
            logger.info(
                "[LZAnimaCrossAttn] strength=%.2f > 1.0, entering extrapolation: "
                "out = base + %.2f * (artist - base).",
                float(strength), float(strength),
            )

        if not normalize_w and n > 1 and combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            if n >= 4:
                raise ValueError(
                    f"[LZAnimaCrossAttn] normalize_weights=False with {n} artists "
                    f"will amplify output ~{n}x, likely breaking. "
                    f"Enable normalize_weights or use combine=concat."
                )
            elif n >= 2:
                logger.warning(
                    "[LZAnimaCrossAttn] normalize_weights=False with %d artists "
                    "(combine=%s) will amplify output ~%dx.",
                    n, combine_mode, n,
                )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        _cleanup_residual_wrappers(dm)

        ok, num_blocks, ctx_dim, msg = _validate(dm)
        if not ok:
            raise ValueError(f"[LZAnimaCrossAttn] Unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError("[LZAnimaCrossAttn] Not an Anima model")

        explicit_blocks = _parse_layer_filter(layer_filter_text, num_blocks)
        if explicit_blocks is not None:
            target_blocks = explicit_blocks
            sb_real, eb_real = target_blocks[0], target_blocks[-1]
        else:
            sb_real = max(0, sb)
            eb_real = num_blocks - 1 if eb < 0 else min(num_blocks - 1, eb)
            if sb_real > eb_real:
                raise ValueError(
                    f"[LZAnimaCrossAttn] start_block={sb_real} > "
                    f"end_block={eb_real} (total {num_blocks})"
                )
            target_blocks = list(range(sb_real, eb_real + 1))

        sigma_range = None
        if use_sigma_range:
            try:
                ms = model.get_model_object("model_sampling")
                s_at_start = float(ms.percent_to_sigma(start_percent))
                s_at_end = float(ms.percent_to_sigma(end_percent))
                lo, hi = sorted([s_at_end, s_at_start])
                sigma_range = (lo, hi)
            except Exception as e:
                logger.warning(
                    "[LZAnimaCrossAttn] Failed to parse sigma range: %s. "
                    "Timestep control disabled.", e
                )
                sigma_range = None

        m = model.clone()

        state = {
            "enabled": bool(enabled),
            "fusion_mode": fusion_mode,
            "combine_mode": combine_mode,
            "strength": float(strength),
            "apply_to_uncond": bool(apply_to_uncond),
            "raws": raws,
            "ids_list": ids_list,
            "w_list": w_list,
            "user_weights": user_weights,
            "normalize_weights": normalize_w,
            "has_explicit_weights": has_explicit_weights,
            "artist_ema_alpha": artist_ema_alpha,
            "lowrank_k": lowrank_k,
            "artist_static_capture": artist_static_capture,
            "static_capture_k": static_capture_k,
            "artist_anchor_q": artist_anchor_q,
            "anchor_seeds_count": anchor_seeds_count,
            "anchor_user_blend": anchor_user_blend,
            "anchor_deep_layer_threshold": anchor_deep_layer_threshold,
            "individuals": None,
            "real_lens": None,
            "dm_ref": dm,
            "sigma_range": sigma_range,
            "current_sigma": None,
            "_ema_cache": {},
            "_ema_last_sigma": None,
            "_static_cache": {},
            "_static_max_sigma": None,
            "_anchor_cache": {},
            "_anchor_cache_key": None,
            "_in_anchor_run": False,
            "_anchor_failed": False,
        }

        if need_sigma_capture:
            prev = m.model_options.get("model_function_wrapper")
            m.set_model_unet_function_wrapper(_make_sigma_capture(state, prev))

        for i in target_blocks:
            inner = _unwrap_cross_attn(dm.blocks[i].cross_attn)
            wrapper = _CrossAttnWrapper(inner, state, i)
            m.add_object_patch(f"diffusion_model.blocks.{i}.cross_attn", wrapper)

        positive_text = artist_pack.get("base_prompt", "")
        return (m, base_cond_out, positive_text)


class LZAnimaArtistNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Artist chain. Separate with commas or newlines.\n"
                        "Example: wlop, sakimichan, krenz\n"
                        "\n"
                        "Supports two weight syntaxes (can coexist):\n"
                        "  1) Bracket syntax (wlop:1.5) - affects CLIP encoding layer, nonlinear\n"
                        "  2) ::weight syntax ::wlop::1.5 - affects cross-attn injection layer, linear\n"
                        "\n"
                        "Default weight=1.0. Range [0.0, 4.0].\n"
                        "::weight and brackets can be stacked: ::(wlop:1.1)::0.8"
                    )
                }),
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE],
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Artist injection strength.\n"
                        "0.0-1.0: interpolation lerp(base, artist, strength)\n"
                        "1.0-4.0: extrapolation base + strength * (artist - base)\n"
                        "Recommended 1.5-2.5 for extrapolation."
                    )
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Base prompt (optional). Recommend format: '<artist>\\n<base_prompt>'"
                }),
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "base_prompt", "positive_text")
    FUNCTION = "process"
    CATEGORY = "LZNodes/Anima"

    def process(self, clip, model, artist_chain, combine_mode, fusion_mode,
                strength, enabled, apply_to_uncond,
                base_prompt="", advanced_options=None):
        pack_node = LZAnimaArtistPack()
        artist_pack, = pack_node.pack(clip, artist_chain, base_prompt)

        cross_node = LZAnimaArtistCrossAttn()
        patched_model, cond, positive_text = cross_node.patch(
            model, artist_pack, combine_mode, fusion_mode,
            strength, enabled, apply_to_uncond, advanced_options,
        )

        return (patched_model, cond, positive_text)


NODE_CLASS_MAPPINGS = {
    "LZAnimaArtistPack": LZAnimaArtistPack,
    "LZAnimaArtistOptions": LZAnimaArtistOptions,
    "LZAnimaArtistCrossAttn": LZAnimaArtistCrossAttn,
    "LZAnimaArtistNode": LZAnimaArtistNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LZAnimaArtistPack": "LZ Anima Artist Pack (Split + Encode)",
    "LZAnimaArtistOptions": "LZ Anima Artist Options (Advanced)",
    "LZAnimaArtistCrossAttn": "LZ Anima Artist Cross-Attn (v2)",
    "LZAnimaArtistNode": "LZ Anima Artist Node",
}
