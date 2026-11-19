import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.phaseII_full_paired_v2 import forward_with_components, patched_logits


def tiny_gpt2():
    config = GPT2Config(
        vocab_size=31,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    return GPT2LMHeadModel(config).eval()


def test_component_capture_shapes_and_finite_values():
    torch.manual_seed(7)
    model = tiny_gpt2()
    ids = torch.tensor([[1, 2, 3, 4]])
    residual, attention, mlp, logits = forward_with_components(model, ids, [1, 2])

    assert len(residual) == len(attention) == len(mlp) == 2
    assert all(x.shape == (16,) for x in residual + attention + mlp)
    assert logits.shape == (31,)
    assert all(torch.isfinite(x).all() for x in residual + attention + mlp + [logits])


def test_patching_own_component_activation_is_identity():
    torch.manual_seed(11)
    model = tiny_gpt2()
    ids = torch.tensor([[1, 2, 3, 4]])
    residual, attention, mlp, logits = forward_with_components(model, ids)

    for layer_idx in (1, 2):
        for component, component_values in (
            ("residual", residual),
            ("attention", attention),
            ("mlp", mlp),
        ):
            patched = patched_logits(
                model, ids, layer_idx, component_values[layer_idx - 1],
                component=component,
            )
            torch.testing.assert_close(patched, logits, rtol=1e-5, atol=1e-6)


def test_hooks_are_removed_after_forward_and_patch():
    torch.manual_seed(13)
    model = tiny_gpt2()
    ids = torch.tensor([[2, 5, 7]])
    _, attention, _, logits = forward_with_components(model, ids)
    patched_logits(model, ids, 1, attention[0], component="attention")
    repeated = model(input_ids=ids, use_cache=False).logits[0, -1, :]
    torch.testing.assert_close(repeated, logits, rtol=1e-5, atol=1e-6)
