"""Mechanism-tool invariants on the real architecture (OLMo-2), using a tiny random OLMo-2 if available."""
import pytest
import torch

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def lm():
    from transformers import Olmo2Config, Olmo2ForCausalLM

    from freqsub.scoring.hf_scorer import LoadedModel

    torch.manual_seed(0)
    cfg = Olmo2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      tie_word_embeddings=False)
    model = Olmo2ForCausalLM(cfg).eval()
    return LoadedModel(model, None, "cpu", torch.float32)


IDS = [5, 17, 33, 2, 99, 64]


def test_dla_decomposition_is_exact(lm):
    from freqsub.interp.components import direct_logit_attribution, run_with_cache
    cache = run_with_cache(lm, IDS)
    a = direct_logit_attribution(lm, cache, target=7, foil=11)
    parts = sum(v for k, v in a.items() if not k.startswith("_"))
    assert abs(parts - a["_total"]) < 1e-4          # embed + components == final residual
    assert abs(a["_total"] - a["_actual"]) < 1e-3   # frozen-norm linearisation reproduces the logit diff


def test_logit_lens_last_layer_matches_model(lm):
    from freqsub.interp.components import logit_lens, run_with_cache
    cache = run_with_cache(lm, IDS)
    lens = logit_lens(lm, cache, 7, 11)
    assert len(lens) == 3
    assert abs(lens[-1] - float(cache["logits"][7] - cache["logits"][11])) < 1e-3


def test_ablation_with_own_value_is_identity(lm):
    from freqsub.interp.components import logits_with_ablation, run_with_cache
    cache = run_with_cache(lm, IDS)
    same = logits_with_ablation(lm, IDS, {"L1.mlp": cache["components"]["L1.mlp"]})
    assert torch.allclose(same, cache["logits"], atol=1e-4)
    changed = logits_with_ablation(lm, IDS, {"L1.mlp": torch.zeros(32)})
    assert not torch.allclose(changed, cache["logits"], atol=1e-4)


def test_prompt_ablation_context_matches_logits_with_ablation(lm):
    """The prompt-pass ablation context must reproduce logits_with_ablation at the answer position."""
    from freqsub.interp.components import ablation_at_last_position, logits_with_ablation, run_with_cache
    cache = run_with_cache(lm, IDS)
    rep = {"L0.attn": torch.zeros(32), "L2.mlp": cache["components"]["L1.mlp"]}
    with torch.no_grad(), ablation_at_last_position(lm, rep):
        lg = lm.model(torch.tensor([IDS])).logits[0, -1].float()
    assert torch.allclose(lg, logits_with_ablation(lm, IDS, rep), atol=1e-4)
