"""
Tests for T2VCausalModel training step correctness.

Verifies:
  1. Each training type (tf, df, tf_dcm, tf_scm, sf_dmd) runs forward+backward without error.
  2. Replayed training produces the same loss as non-replayed training.
  3. Gradient comparison between replayed and non-replayed paths.

Usage:
    pytest -s rcm/models/t2v_model_causal_test.py
"""

import pytest
import torch
from einops import repeat

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate as lazy_instantiate
from rcm.conditioner import DataType, TextCondition
from rcm.models.t2v_model_causal import T2VCausalConfig, T2VCausalModel
from rcm.networks.wan2pt1 import WanModel
from rcm.networks.wan2pt1_jvp import WanModel_JVP
from rcm.utils.blockmask import AttnMaskSpec
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode
from rcm.utils.timestep_utils import LogNormal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for causal model tests")

B, C, T, H, W = 2, 16, 5, 8, 8
TEXT_LEN = 64
TEXT_DIM = 64

_MINI_NET_KWARGS = dict(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=TEXT_LEN,
    in_dim=C,
    dim=128,
    ffn_dim=256,
    freq_dim=256,
    text_dim=TEXT_DIM,
    out_dim=C,
    num_heads=4,
    num_layers=2,
)

MINI_NET = L(WanModel)(**_MINI_NET_KWARGS)
MINI_NET_JVP = L(WanModel_JVP)(**_MINI_NET_KWARGS)


def _build_test_model(
    training_type="tf",
    with_jvp_net=False,
    with_causal_teacher=False,
    with_bidirectional_teacher=False,
    with_fake_score=False,
    **config_overrides,
):
    base_kwargs = dict(
        state_ch=C,
        state_t=T,
        first_chunk_t=1,
        chunk_t=1,
        training_type=training_type,
        replayed_training=False,
        precision="float32",
        loss_scale=1.0,
        causal_guidance=1.0,
        bidirectional_guidance=1.0,
        tangent_warmup=0,
    )
    base_kwargs.update(config_overrides)
    config = T2VCausalConfig(**base_kwargs)

    model = T2VCausalModel.__new__(T2VCausalModel)
    torch.nn.Module.__init__(model)

    model.config = config
    model.precision = torch.float32
    model.tensor_kwargs = {"device": "cuda", "dtype": torch.float32}
    model.p_G = LogNormal(p_mean=-0.8, p_std=1.6)
    model.p_D = LogNormal(p_mean=0.0, p_std=1.6)
    model.scaling = RectifiedFlow_TrigFlowWrapper(config.sigma_data, config.rectified_flow_t_scaling_factor)
    model.neg_embed = None
    model.fsdp_device_mesh = None
    model.data_parallel_size = 1
    model.tokenizer = None
    model.conditioner = None

    net_cfg = MINI_NET_JVP if with_jvp_net else MINI_NET
    model.net = lazy_instantiate(net_cfg).cuda()

    if with_causal_teacher:
        model.net_causal_teacher = lazy_instantiate(MINI_NET).cuda()
        model.net_causal_teacher.eval().requires_grad_(False)
    else:
        model.net_causal_teacher = None

    if with_bidirectional_teacher:
        model.net_bidirectional_teacher = lazy_instantiate(MINI_NET).cuda()
        model.net_bidirectional_teacher.eval().requires_grad_(False)
    else:
        model.net_bidirectional_teacher = None

    if with_fake_score:
        model.net_fake_score = lazy_instantiate(MINI_NET).cuda()
        model.net_fake_score.eval().requires_grad_(False)
    else:
        model.net_fake_score = None

    return model


def _make_synthetic_data():
    """Create deterministic synthetic data batch and conditions."""
    x0 = torch.randn(B, C, T, H, W, device="cuda")
    condition = TextCondition(
        crossattn_emb=torch.randn(B, TEXT_LEN, TEXT_DIM, device="cuda"),
        data_type=DataType.VIDEO,
    )
    uncondition = TextCondition(
        crossattn_emb=torch.randn(B, TEXT_LEN, TEXT_DIM, device="cuda"),
        data_type=DataType.VIDEO,
    )
    return x0, condition, uncondition


def _mock_get_data_and_condition(x0, condition, uncondition):

    def _fn(data_batch):
        return None, x0.clone(), condition, uncondition

    return _fn


def _collect_grads(module):
    return {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


def _assert_matching_grads(grads_nr, grads_r, *, rtol=1e-3, atol=1e-3):
    assert set(grads_nr) == set(grads_r), f"Gradient key mismatch: {set(grads_nr) ^ set(grads_r)}"
    for name, grad_nr in grads_nr.items():
        try:
            torch.testing.assert_close(grad_nr, grads_r[name], rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc


def _make_cached_prefix_state(model, net, x0, condition, block_index):
    _, _, _, _, _, frame_tokens, _, block_pattern = model._make_block_pattern(x0.size(), model=net)
    kv_caches = net.allocate_kv_caches(max_len=x0.shape[2] * frame_tokens)
    zeros = torch.zeros(x0.shape[0], 1, device=x0.device, dtype=x0.dtype)
    for i in range(block_index):
        frame_start, frame_end, block_size = model._block_span(block_pattern, i)
        time_block = repeat(zeros, "b 1 -> b t", t=block_size)
        state = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_caches, pattern=block_pattern, block_cursor=i)
        net(
            x0[:, :, frame_start:frame_end],
            time_block,
            **condition.to_dict(),
            inference_state=state,
            attn_meta=AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=i),
        )
    return kv_caches, block_pattern


@pytest.mark.L1
def test_full_prefix_cached_forward_matches_direct_forward():
    model = _build_test_model(training_type="tf")
    x0, cond, _ = _make_synthetic_data()
    block_index = 2
    kv_caches, block_pattern = _make_cached_prefix_state(model, model.net, x0, cond, block_index)
    frame_start, frame_end, block_size = model._block_span(block_pattern, block_index)
    zeros_block = torch.zeros(B, block_size, device="cuda")

    cached = model.net(
        x0[:, :, frame_start:frame_end],
        zeros_block,
        **cond.to_dict(),
        inference_state=CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv_caches, pattern=block_pattern, block_cursor=block_index),
        attn_meta=AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=block_index),
    )
    direct = model.net(
        x0[:, :, :frame_end],
        torch.zeros(B, frame_end, device="cuda"),
        **cond.to_dict(),
    )[:, :, frame_start:frame_end]

    torch.testing.assert_close(cached, direct, rtol=1e-4, atol=1e-4)


@pytest.mark.L1
def test_full_prefix_cached_jvp_forward_matches_direct_forward():
    model = _build_test_model(training_type="tf_scm", with_jvp_net=True)
    x0, cond, _ = _make_synthetic_data()
    block_index = 2
    kv_caches, block_pattern = _make_cached_prefix_state(model, model.net, x0, cond, block_index)
    frame_start, frame_end, block_size = model._block_span(block_pattern, block_index)
    zeros_block = torch.zeros(B, block_size, device="cuda")

    cached = model.net(
        x0[:, :, frame_start:frame_end],
        zeros_block,
        **cond.to_dict(),
        inference_state=CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv_caches, pattern=block_pattern, block_cursor=block_index),
        attn_meta=AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=block_index),
    )
    direct = model.net(
        x0[:, :, :frame_end],
        torch.zeros(B, frame_end, device="cuda"),
        **cond.to_dict(),
    )[:, :, frame_start:frame_end]

    torch.testing.assert_close(cached, direct, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 1: each training type runs forward + backward
# ---------------------------------------------------------------------------


@pytest.mark.L1
@pytest.mark.parametrize(
    "training_type, needs_teacher",
    [("tf", False), ("df", False), ("tf_dcm", True)],
)
def test_training_runs(training_type, needs_teacher):
    model = _build_test_model(training_type=training_type, with_causal_teacher=needs_teacher)
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)
    model.net.train()

    total_loss = torch.zeros(B, device="cuda")
    n_closures = 0
    for name, closure, is_last in model.training_step_closures({}, iteration=0):
        out, loss = closure()
        assert loss.shape == (B,), f"Loss shape mismatch: {loss.shape}"
        assert torch.isfinite(loss).all(), f"Non-finite loss in closure '{name}'"
        total_loss = total_loss + loss
        n_closures += 1
    assert n_closures >= 1, f"Expected at least 1 closure for {training_type}"
    assert is_last, "Last closure should have is_last=True"

    total_loss.sum().backward()

    has_grad = any(p.grad is not None for p in model.net.parameters())
    assert has_grad, "No gradients computed for model.net"


@pytest.mark.L1
def test_tf_scm_training_runs():
    model = _build_test_model(
        training_type="tf_scm",
        with_jvp_net=True,
        with_causal_teacher=True,
    )
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)
    model.net.train()

    total_loss = torch.zeros(B, device="cuda")
    n_closures = 0
    for name, closure, is_last in model.training_step_closures({}, iteration=0):
        out, loss = closure()
        assert loss.shape == (B,), f"Loss shape mismatch: {loss.shape}"
        total_loss = total_loss + loss
        n_closures += 1
    assert n_closures >= 1
    assert is_last

    total_loss.sum().backward()
    assert any(p.grad is not None for p in model.net.parameters())


@pytest.mark.L1
def test_sf_dmd_training_runs():
    model = _build_test_model(
        training_type="sf_dmd",
        with_bidirectional_teacher=True,
        with_fake_score=True,
    )
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)

    total_loss = torch.zeros(B, device="cuda")
    n_closures = 0
    for name, closure, is_last in model.training_step_closures({}, iteration=0):
        out, loss = closure()
        assert loss.shape == (B,), f"Loss shape mismatch: {loss.shape}"
        total_loss = total_loss + loss
        n_closures += 1
    assert n_closures >= 1
    assert is_last

    total_loss.sum().backward()
    assert any(p.grad is not None for p in model.net.parameters())


# ---------------------------------------------------------------------------
# Test 2: SF-DMD replayed vs non-replayed
# ---------------------------------------------------------------------------


@pytest.mark.L1
def test_sf_dmd_replayed_matches_nonreplayed():
    model = _build_test_model(
        training_type="sf_dmd",
        with_bidirectional_teacher=True,
        with_fake_score=True,
    )
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)
    model.net.eval()
    model.net_bidirectional_teacher.eval()
    model.net_fake_score.eval()

    # --- non-replayed (seed works: both paths use same random op order) ---
    model.net.zero_grad()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    _, loss_nr = model._sf_step({}, iteration=0)
    loss_nr.sum().backward()
    grads_nr = _collect_grads(model.net)

    # --- replayed ---
    model.net.zero_grad()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_blocks, ctx = model._make_replayed_sf_ctx({}, iteration=0)
    assert num_blocks == T

    total_loss_r = torch.zeros(B, device="cuda")
    for i in range(num_blocks):
        _, loss_i = model._replayed_sf_step(i, ctx, iteration=0)
        total_loss_r = total_loss_r + loss_i
    total_loss_r.sum().backward()
    grads_r = _collect_grads(model.net)

    torch.testing.assert_close(loss_nr, total_loss_r, rtol=1e-4, atol=1e-4)

    for name in grads_nr:
        assert name in grads_r, f"Missing gradient for {name} in replayed path"


@pytest.mark.L1
def test_sf_dmd_per_chunk_steps_replayed_matches_nonreplayed():
    model = _build_test_model(
        training_type="sf_dmd",
        with_bidirectional_teacher=True,
        with_fake_score=True,
        sf_simulation_steps_per_chunk=[4, 2],
        sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], [5 / 6]],
        student_update_freq=1,
    )
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)
    model.net.eval()
    model.net_bidirectional_teacher.eval()
    model.net_fake_score.eval()

    fixed_step_counts = [4, 2, 2, 2, 2]
    model._self_forcing_simulation_steps = lambda iteration, fake_score_update=False, num_blocks=None: fixed_step_counts[:num_blocks]
    critic_model = _build_test_model(
        training_type="sf_dmd",
        with_bidirectional_teacher=True,
        with_fake_score=True,
        sf_simulation_steps_per_chunk=[4, 2],
    )
    critic_model.get_effective_iteration_fake = lambda iteration: iteration - 1
    critic_model.get_effective_iteration = lambda iteration: iteration - 1
    assert critic_model._self_forcing_simulation_steps(iteration=1, fake_score_update=True, num_blocks=T) == [1, 1, 1, 1, 1]
    assert critic_model._self_forcing_simulation_steps(iteration=4, fake_score_update=True, num_blocks=T) == [4, 2, 2, 2, 2]
    assert critic_model._self_forcing_simulation_steps(iteration=5, fake_score_update=True, num_blocks=T) == [1, 1, 1, 1, 1]
    critic_model.config.sf_simulation_steps_per_chunk = [4, 5]
    try:
        critic_model._self_forcing_simulation_steps(iteration=1, fake_score_update=True, num_blocks=T)
        raise AssertionError("Expected increasing per-chunk max steps to fail")
    except ValueError:
        pass
    del critic_model

    model.net.zero_grad()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    _, loss_nr = model._sf_step({}, iteration=63)
    loss_nr.sum().backward()
    grads_nr = _collect_grads(model.net)

    model.net.zero_grad()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_blocks, ctx = model._make_replayed_sf_ctx({}, iteration=63)
    assert num_blocks == T
    _, G_time_B_T, _, _, _ = ctx
    expected_time = torch.tensor(
        [5 / 8, 5 / 6, 5 / 6, 5 / 6, 5 / 6],
        device="cuda",
    ).expand(B, T)
    torch.testing.assert_close(G_time_B_T, expected_time)

    total_loss_r = torch.zeros(B, device="cuda")
    for i in range(num_blocks):
        _, loss_i = model._replayed_sf_step(i, ctx, iteration=63)
        total_loss_r = total_loss_r + loss_i
    total_loss_r.sum().backward()
    grads_r = _collect_grads(model.net)

    torch.testing.assert_close(loss_nr, total_loss_r, rtol=1e-4, atol=1e-4)

    for name in grads_nr:
        assert name in grads_r, f"Missing gradient for {name} in replayed path"


@pytest.mark.L1
def test_sf_dmd_context_from_last_step_backward_runs():
    model = _build_test_model(
        training_type="sf_dmd",
        with_bidirectional_teacher=True,
        with_fake_score=True,
        context_from_last_step=True,
        sf_simulation_steps_per_chunk=[4, 2],
        sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], [5 / 8]],
        student_update_freq=1,
    )
    x0, cond, uncond = _make_synthetic_data()
    model.get_data_and_condition = _mock_get_data_and_condition(x0, cond, uncond)
    model.net.eval()
    model.net_bidirectional_teacher.eval()
    model.net_fake_score.eval()

    _, loss = model._sf_step({}, iteration=63)
    loss.sum().backward()
    assert any(p.grad is not None for p in model.net.parameters())
