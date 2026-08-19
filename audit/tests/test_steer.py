"""Hook semantics, pinned against `activation_steer.ActivationSteerer` itself.

Everything here runs on a few-hundred-parameter model on the CPU: no GPU, no
network, no keys. The point is not that steering *works* — it is that this
module's hook site, dtype order and position policy are the ones upstream uses,
which is a property of tiny tensors, not of a 7B model.

Skipped whole if torch is absent: the rest of the audit suite runs without it,
and a forward hook cannot be exercised against a stub. The `generate()` tests at
the end additionally need `transformers` and skip without it.
"""

import csv
import pathlib
import types

import pytest

torch = pytest.importorskip("torch")

from audit import elicit, generate, steer  # noqa: E402  (after the skip guard)
from audit.tests import fakes  # noqa: E402

HIDDEN = 8
LAYERS = 4
#: upstream's 1-based --layer, so the layer list is indexed at 2
LAYER = 3
VECTOR_ROWS = LAYERS + 1


# --- a model small enough to be exact -----------------------------------------

class TupleBlock(torch.nn.Module):
    """A decoder block as transformers 4.52 writes one: returns `(hidden, ...)`."""

    def __init__(self, hidden, extra=None):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden)
        self.extra = extra

    def forward(self, hidden_states):
        out = self.linear(hidden_states)
        return (out, self.extra) if self.extra is not None else (out,)


class TensorBlock(TupleBlock):
    """A decoder block as newer transformers writes one: returns a bare tensor."""

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class Inner(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class TinyModel(torch.nn.Module):
    """`model.model.layers[...]` — the path Qwen2 exposes and upstream finds third."""

    def __init__(self, block=TupleBlock, extra=None, dtype=torch.float32):
        super().__init__()
        self.model = Inner([block(HIDDEN, extra) for _ in range(LAYERS)])
        self.config = types.SimpleNamespace(hidden_size=HIDDEN)
        self.to(dtype)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyModel()


@pytest.fixture
def vector_file(tmp_path):
    """A vector file shaped like the real one: rows indexed by layer, float32."""
    torch.manual_seed(1)
    path = tmp_path / "trait_response_avg_diff.pt"
    torch.save(torch.randn(VECTOR_ROWS, HIDDEN), path)
    return path


@pytest.fixture
def steering(vector_file):
    return steer.Steering(str(vector_file), LAYER, 2.0)


def hidden(batch=2, width=5, dtype=torch.float32):
    torch.manual_seed(2)
    return torch.randn(batch, width, HIDDEN, dtype=dtype)


# --- the hook site --------------------------------------------------------------

def test_hook_site_is_the_one_upstream_hooks(model, steering):
    """Not "the same index" — the same object, from their own search."""
    upstream = pytest.importorskip("activation_steer")
    theirs = upstream.ActivationSteerer(
        model, torch.zeros(HIDDEN), coeff=0.0, layer_idx=LAYER - 1,
        positions="all")._locate_layer()
    path, ours = steer.hooked_module(model, LAYER)
    assert ours is theirs
    assert path == "model.layers[%d]" % (LAYER - 1)


def test_layer_indexes_the_list_at_layer_minus_one(model):
    _path, module = steer.hooked_module(model, LAYER)
    assert module is model.model.layers[LAYER - 1]


def test_upstream_search_order_is_reproduced():
    """`transformer.h` wins over `model.layers` when a model has both — as theirs does."""
    both = TinyModel()
    both.transformer = types.SimpleNamespace(h=["gpt2-style"])
    path, layers = steer.upstream_layer_list(both)
    assert (path, layers) == ("transformer.h", ["gpt2-style"])


def test_layer_out_of_range_is_refused(model, vector_file):
    with pytest.raises(steer.SteeringError, match="has %d layers" % LAYERS):
        steer.hooked_module(model, LAYERS + 1)


def test_no_layer_list_is_refused(vector_file):
    with pytest.raises(steer.SteeringError, match="no layer list"):
        steer.upstream_layer_list(types.SimpleNamespace())


# --- beta = 0 is a no-op --------------------------------------------------------

def assert_noop_with_controls(model, vector_file, activations):
    """coeff 0 changes nothing, plus the controls that make that mean something.

    Deliberately one test and not four. The no-op assertion on its own also passes
    when `__enter__` never registers the hook, so the proof that a hook WAS
    installed and that a nonzero coefficient DOES move the output has to fail in
    the same test, not in a sibling that could stay green.
    """
    block = model.model.layers[LAYER - 1]
    with torch.no_grad():
        unhooked = block(activations)[0]

    zero = steer.Steering(str(vector_file), LAYER, 0.0)
    with steer.ActivationSteering(model, zero), torch.no_grad():
        assert len(block._forward_hooks) == 1, "no hook was installed; the check is void"
        at_zero = block(activations)[0]
    assert torch.equal(at_zero, unhooked)

    nonzero = steer.Steering(str(vector_file), LAYER, 2.0)
    with steer.ActivationSteering(model, nonzero) as installed, torch.no_grad():
        assert len(block._forward_hooks) == 1
        moved = block(activations)[0]
    assert torch.equal(moved, unhooked + installed.delta)
    assert not torch.equal(moved, unhooked), "the hook changed nothing; the check is void"


def test_coeff_zero_is_bit_identical_to_no_hook(model, vector_file):
    assert_noop_with_controls(model, vector_file, hidden())


def test_coeff_zero_is_bit_identical_in_bfloat16(vector_file):
    assert_noop_with_controls(TinyModel(dtype=torch.bfloat16), vector_file,
                              hidden(dtype=torch.bfloat16))


def test_coeff_zero_is_bit_identical_at_a_width_one_decode_step(model, vector_file):
    """The decode-step shape, which is where a positions bug would actually bite."""
    assert_noop_with_controls(model, vector_file, hidden(width=1))


def test_hook_is_removed_on_exit(model, steering):
    block = model.model.layers[LAYER - 1]
    with steer.ActivationSteering(model, steering):
        assert len(block._forward_hooks) == 1
    assert len(block._forward_hooks) == 0


def test_hook_is_removed_when_the_body_raises(model, steering):
    block = model.model.layers[LAYER - 1]
    with pytest.raises(ZeroDivisionError):
        with steer.ActivationSteering(model, steering):
            raise ZeroDivisionError
    assert len(block._forward_hooks) == 0


def test_double_install_is_refused(model, steering):
    with steer.ActivationSteering(model, steering) as installed:
        with pytest.raises(steer.SteeringError, match="already installed"):
            installed.__enter__()


# --- what gets added, and where -------------------------------------------------

def test_all_positions_adds_the_delta_at_every_position(model, steering):
    block = model.model.layers[LAYER - 1]
    activations = hidden()
    with torch.no_grad():
        unhooked = block(activations)[0]
    with steer.ActivationSteering(model, steering) as installed, torch.no_grad():
        hooked = block(activations)[0]
    # compared against the sum, not against the difference: (x + d) - x is not d
    assert torch.equal(hooked, unhooked + installed.delta)


def test_all_positions_adds_at_a_width_one_decode_step(model, steering):
    """`all` means all: a decode step is one position, not an exemption."""
    block = model.model.layers[LAYER - 1]
    decode_step = hidden(width=1)
    with torch.no_grad():
        unhooked = block(decode_step)[0]
    with steer.ActivationSteering(model, steering) as installed, torch.no_grad():
        hooked = block(decode_step)[0]
    assert hooked.shape == (2, 1, HIDDEN)
    assert torch.equal(hooked, unhooked + installed.delta)


def test_prompt_positions_adds_when_the_forward_is_wider_than_one(model, vector_file):
    """The other half of their tell — only the skip branch used to be covered."""
    prompt_only = steer.Steering(str(vector_file), LAYER, 2.0, positions="prompt")
    block = model.model.layers[LAYER - 1]
    activations = hidden(width=5)
    with torch.no_grad():
        unhooked = block(activations)[0]
    with steer.ActivationSteering(model, prompt_only) as installed, torch.no_grad():
        hooked = block(activations)[0]
    assert torch.equal(hooked, unhooked + installed.delta)


def test_prompt_positions_skips_a_width_one_forward(model, vector_file):
    prompt_only = steer.Steering(str(vector_file), LAYER, 2.0, positions="prompt")
    block = model.model.layers[LAYER - 1]
    decode_step = hidden(width=1)
    with torch.no_grad():
        unhooked = block(decode_step)[0]
    with steer.ActivationSteering(model, prompt_only), torch.no_grad():
        hooked = block(decode_step)[0]
    assert torch.equal(hooked, unhooked)


def test_response_positions_touches_only_the_last(model, vector_file):
    last_only = steer.Steering(str(vector_file), LAYER, 2.0, positions="response")
    block = model.model.layers[LAYER - 1]
    activations = hidden()
    with torch.no_grad():
        unhooked = block(activations)[0]
    with steer.ActivationSteering(model, last_only) as installed, torch.no_grad():
        hooked = block(activations)[0]
    expected = unhooked.clone()
    expected[:, -1, :] += installed.delta
    assert torch.equal(hooked, expected)
    assert torch.equal(hooked[:, :-1, :], unhooked[:, :-1, :])


def test_extra_tuple_entries_are_carried_through(vector_file):
    model = TinyModel(extra="attention-weights")
    steering = steer.Steering(str(vector_file), LAYER, 2.0)
    with steer.ActivationSteering(model, steering), torch.no_grad():
        out = model.model.layers[LAYER - 1](hidden())
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1] == "attention-weights"


def test_a_bare_tensor_output_is_steered(vector_file):
    model = TinyModel(block=TensorBlock)
    steering = steer.Steering(str(vector_file), LAYER, 2.0)
    block = model.model.layers[LAYER - 1]
    activations = hidden()
    with torch.no_grad():
        unhooked = block(activations)
    with steer.ActivationSteering(model, steering) as installed, torch.no_grad():
        hooked = block(activations)
    assert torch.is_tensor(hooked)
    assert torch.equal(hooked, unhooked + installed.delta)


# --- the dtype order ------------------------------------------------------------

def test_delta_is_cast_before_it_is_scaled(vector_file):
    """`coeff * bfloat16(v)`, not `bfloat16(coeff * v)` — upstream's order.

    The two disagree in the last mantissa bit, which is exactly the kind of
    difference that never shows up in a mean and always shows up in a diff.
    """
    model = TinyModel(dtype=torch.bfloat16)
    coeff = 2.7
    steering = steer.Steering(str(vector_file), LAYER, coeff)
    installed = steer.ActivationSteering(model, steering)
    stored = steering.load_vector()
    assert installed.delta.dtype == torch.bfloat16
    assert torch.equal(installed.delta, coeff * stored.to(torch.bfloat16))
    scaled_first = (coeff * stored).to(torch.bfloat16)
    assert not torch.equal(installed.delta, scaled_first), (
        "this vector no longer distinguishes the two orders; pick another")


def test_delta_matches_upstreams_own_arithmetic(vector_file):
    upstream = pytest.importorskip("activation_steer")
    model = TinyModel(dtype=torch.bfloat16)
    coeff = 2.7
    stored = steer.Steering(str(vector_file), LAYER, coeff).load_vector()
    theirs = upstream.ActivationSteerer(model, stored, coeff=coeff,
                                        layer_idx=LAYER - 1, positions="all")
    ours = steer.ActivationSteering(model, steer.Steering(str(vector_file), LAYER, coeff))
    assert torch.equal(ours.delta, coeff * theirs.vector)


def test_the_vector_is_copied_not_aliased(model, vector_file):
    source = torch.zeros(HIDDEN, requires_grad=True)
    installed = steer.ActivationSteering(
        model, steer.Steering(str(vector_file), LAYER, 1.0), vector=source)
    assert installed.vector is not source
    assert not installed.vector.requires_grad


# --- vector loading and validation ----------------------------------------------

def test_load_vector_takes_the_layer_indexed_row(vector_file):
    stored = torch.load(vector_file, weights_only=False)
    assert torch.equal(steer.Steering(str(vector_file), LAYER, 1.0).load_vector(),
                       stored[LAYER])


def test_a_layer_beyond_the_vector_file_is_refused(vector_file):
    steering = steer.Steering(str(vector_file), VECTOR_ROWS, 1.0)
    with pytest.raises(steer.SteeringError, match="out of range"):
        steering.load_vector()


def test_a_vector_of_the_wrong_width_is_refused(model, vector_file):
    with pytest.raises(steer.SteeringError, match="hidden_size"):
        steer.ActivationSteering(model, steer.Steering(str(vector_file), LAYER, 1.0),
                                 vector=torch.zeros(HIDDEN + 1))


def test_a_non_vector_file_is_refused(tmp_path):
    path = tmp_path / "not_a_vector.pt"
    torch.save(torch.randn(HIDDEN), path)
    with pytest.raises(steer.SteeringError, match="indexed by layer"):
        steer.Steering(str(path), LAYER, 1.0).load_vector()


@pytest.mark.parametrize("layer", [0, -1])
def test_a_layer_below_one_is_refused(vector_file, layer):
    with pytest.raises(steer.SteeringError, match="must be >= 1"):
        steer.Steering(str(vector_file), layer, 1.0)


def test_an_unknown_position_policy_is_refused(vector_file):
    with pytest.raises(steer.SteeringError, match="positions must be one of"):
        steer.Steering(str(vector_file), LAYER, 1.0, positions="last")


def test_a_missing_vector_file_is_refused(tmp_path):
    with pytest.raises(steer.SteeringError, match="no steering vector at"):
        steer.Steering(str(tmp_path / "absent.pt"), LAYER, 1.0)


def test_vector_sha256_fingerprints_the_file(vector_file):
    import hashlib

    steering = steer.Steering(str(vector_file), LAYER, 1.0)
    expected = hashlib.sha256(vector_file.read_bytes()).hexdigest()[:16]
    assert steering.vector_sha256 == expected


# --- the engine wrapper ----------------------------------------------------------

class ProbeEngine:
    """Records how many hooks were live on the block while it generated."""

    def __init__(self, block, continuation="12"):
        self._block = block
        self._continuation = continuation
        self.hooks_while_generating = []
        self.seeds = []

    def describe(self):
        return fakes.FAKE_ENGINE_DESCRIPTION

    def generate(self, prompts, sampling, seed):
        self.hooks_while_generating.append(len(self._block._forward_hooks))
        self.seeds.append(seed)
        return [self._continuation] * len(prompts)


def test_the_hook_is_live_for_the_whole_batch_and_gone_after(model, steering):
    block = model.model.layers[LAYER - 1]
    engine = ProbeEngine(block)
    steered = steer.SteeredEngine(engine, model, steering)
    steered.generate(["a", "b"], object(), 7)
    assert engine.hooks_while_generating == [1]
    assert len(block._forward_hooks) == 0


def test_a_bad_steering_fails_before_any_generation(model, vector_file):
    engine = ProbeEngine(model.model.layers[0])
    with pytest.raises(steer.SteeringError):
        steer.SteeredEngine(engine, model, steer.Steering(str(vector_file), LAYERS + 1, 1.0))
    assert engine.hooks_while_generating == []


def test_the_engine_description_passes_through(model, steering):
    steered = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model, steering)
    assert steered.describe() == fakes.FAKE_ENGINE_DESCRIPTION


def test_steering_columns_report_what_ran(model, steering, vector_file):
    columns = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model,
                                  steering).columns
    assert set(columns) == set(steer.STEERING_FIELDS)
    assert columns["steer_coeff"] == 2.0
    assert columns["steer_layer"] == LAYER
    assert columns["steer_module_index"] == LAYER - 1
    assert columns["steer_module_path"] == "model.layers[%d]" % (LAYER - 1)
    assert columns["steer_positions"] == "all"
    assert columns["steer_vector_sha256"] == steering.vector_sha256
    assert columns["steer_delta_dtype"] == "torch.float32"


# --- rows ------------------------------------------------------------------------

def a_row(**overrides):
    fields = {name: "" for name in generate.ROW_FIELDS}
    fields.update(sample_index=0, batch_index=0, batch_size=1, seed=0, value=None,
                  temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                  repetition_penalty=1.0, max_new_tokens=1000, min_new_tokens=1,
                  repo_dirty=False)
    fields.update(overrides)
    return generate.Row(**fields)


def test_the_steering_columns_are_written_on_every_row(tmp_path, model, steering):
    columns = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model,
                                  steering).columns
    path = tmp_path / "rows.csv"
    with steer.SteeredRowWriter(path, columns) as writer:
        assert writer.write([a_row(value=12.0), a_row(value=None, tag="unresolved")]) == 2
    with path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == list(generate.ROW_FIELDS + steer.STEERING_FIELDS)
    assert [row["steer_coeff"] for row in written] == ["2.0", "2.0"]
    # an unresolved answer keeps its row and its tag, and never becomes a zero
    assert written[1]["value"] == "" and written[1]["tag"] == "unresolved"


def test_rows_reach_disk_before_the_run_ends(tmp_path, model, steering):
    columns = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model,
                                  steering).columns
    path = tmp_path / "rows.csv"
    with steer.SteeredRowWriter(path, columns) as writer:
        writer.write([a_row(value=1.0)])
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_writer_refuses_columns_it_cannot_describe(tmp_path):
    with pytest.raises(steer.SteeringError, match="missing"):
        steer.SteeredRowWriter(tmp_path / "rows.csv", {"steer_coeff": 1.0})


def test_the_writer_refuses_anything_that_is_not_a_row(tmp_path, model, steering):
    columns = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model,
                                  steering).columns
    with steer.SteeredRowWriter(tmp_path / "rows.csv", columns) as writer:
        with pytest.raises(TypeError, match="takes Row objects"):
            writer.write([{"value": 1.0}])


def test_a_closed_writer_refuses_to_write(tmp_path, model, steering):
    columns = steer.SteeredEngine(ProbeEngine(model.model.layers[0]), model,
                                  steering).columns
    writer = steer.SteeredRowWriter(tmp_path / "rows.csv", columns)
    with writer:
        pass
    with pytest.raises(RuntimeError, match="closed"):
        writer.write([a_row()])


def test_rows_path_names_the_coefficient(tmp_path, vector_file):
    for coeff, label in ((-5.0, "-5"), (0.0, "0"), (2.5, "2.5")):
        steering = steer.Steering(str(vector_file), 20, coeff)
        path = steer.rows_path(tmp_path, steering, "altruism_v3/dictator", "free")
        assert path.name == "altruism_v3-dictator_free_layer20_all_coef%s.csv" % label


# --- the sweep --------------------------------------------------------------------

def test_a_sweep_writes_one_file_per_coefficient(tmp_path, model, vector_file):
    engine = ProbeEngine(model.model.layers[LAYER - 1], continuation="I will give $12.")
    sampling = generate.NEUTRAL.sampling
    steerings = [steer.Steering(str(vector_file), LAYER, coeff) for coeff in (-1.0, 0.0, 1.0)]
    seen = []
    results = steer.sweep([("altruism_v3/dictator", "free")], engine, model, steerings,
                          sampling, fakes.QwenLikeTokenizer(), samples_per_prompt=4,
                          reading="stated", batch_size=2, out_dir=tmp_path,
                          on_coefficient=lambda s, p, r: seen.append(s.coeff))
    assert seen == [-1.0, 0.0, 1.0]
    assert sorted(results) == [-1.0, 0.0, 1.0]
    for coeff, (path, rows) in results.items():
        assert len(rows) == 4 and path.exists()
        with path.open(encoding="utf-8", newline="") as handle:
            written = list(csv.DictReader(handle))
        assert len(written) == 4
        assert {row["steer_coeff"] for row in written} == {str(coeff)}
        assert {row["value"] for row in written} == {"12.0"}
    # the hook was live for every batch of every coefficient: 3 coefficients x 2 batches
    assert engine.hooks_while_generating == [1] * 6


def test_every_coefficient_sees_the_same_prompts_and_seeds(tmp_path, model, vector_file):
    engine = ProbeEngine(model.model.layers[LAYER - 1], continuation="I will give $12.")
    steerings = [steer.Steering(str(vector_file), LAYER, coeff) for coeff in (-1.0, 1.0)]
    steer.sweep([("altruism_v3/dictator", "free")], engine, model, steerings,
                generate.NEUTRAL.sampling, fakes.QwenLikeTokenizer(),
                samples_per_prompt=4, reading="stated", batch_size=2, out_dir=tmp_path)
    assert engine.seeds == [0, 1, 0, 1]


def test_a_repeated_coefficient_is_refused(tmp_path, model, vector_file):
    steerings = [steer.Steering(str(vector_file), LAYER, 1.0)] * 2
    with pytest.raises(steer.SteeringError, match="appears twice"):
        steer.sweep([("altruism_v3/dictator", "free")],
                    ProbeEngine(model.model.layers[0]), model, steerings,
                    generate.NEUTRAL.sampling, fakes.QwenLikeTokenizer(),
                    samples_per_prompt=1, reading="stated", batch_size=1,
                    out_dir=tmp_path)


def test_a_sweep_with_no_games_is_refused(tmp_path, model, vector_file):
    with pytest.raises(steer.SteeringError, match="at least one .game_id, mode. pair"):
        steer.sweep([], ProbeEngine(model.model.layers[0]), model,
                    [steer.Steering(str(vector_file), LAYER, 1.0)],
                    generate.NEUTRAL.sampling, fakes.QwenLikeTokenizer(),
                    samples_per_prompt=1, reading="stated", batch_size=1,
                    out_dir=tmp_path)


def test_an_empty_sweep_is_refused(tmp_path, model):
    with pytest.raises(steer.SteeringError, match="at least one steering"):
        steer.sweep([("altruism_v3/dictator", "free")],
                    ProbeEngine(model.model.layers[0]), model, [],
                    generate.NEUTRAL.sampling, fakes.QwenLikeTokenizer(),
                    samples_per_prompt=1, reading="stated", batch_size=1,
                    out_dir=tmp_path)


# --- the projection cross-check ---------------------------------------------------

class HiddenStateModel:
    """Returns fixed hidden states, so the projection arithmetic is checkable by hand."""

    device = "cpu"

    def __init__(self, states):
        self._states = states

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        assert output_hidden_states
        return types.SimpleNamespace(hidden_states=self._states)


class CharTokenizer:
    """One id per character, and no merging across a concatenation."""

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def encode(self, text, add_special_tokens=None):
        return [ord(c) for c in text]


class MergingTokenizer(CharTokenizer):
    """Merges the pair straddling the seam, so the prompt's ids stop being a prefix."""

    def encode(self, text, add_special_tokens=None):
        return [ord(character) for character in text.replace("!d", "@")]


def test_projection_is_the_scalar_projection_of_the_answer_average():
    prompt, answer = "abc", "de"
    width = len(prompt) + len(answer)
    states = [torch.zeros(1, width, HIDDEN) for _ in range(LAYER + 1)]
    states[LAYER] = torch.arange(width * HIDDEN, dtype=torch.float32).reshape(1, width, HIDDEN)
    vector = torch.arange(HIDDEN, dtype=torch.float32)
    got = steer.response_avg_projection(HiddenStateModel(states), CharTokenizer(),
                                        prompt, answer, vector, LAYER)
    expected_avg = states[LAYER][:, len(prompt):, :].mean(dim=1)
    expected = ((expected_avg * vector).sum(dim=-1) / vector.norm()).item()
    assert got == pytest.approx(expected)


def test_projection_reads_the_layer_it_is_given():
    prompt, answer = "abc", "de"
    width = len(prompt) + len(answer)
    states = [torch.full((1, width, HIDDEN), float(i)) for i in range(LAYER + 2)]
    vector = torch.ones(HIDDEN)
    at_layer = steer.response_avg_projection(HiddenStateModel(states), CharTokenizer(),
                                             prompt, answer, vector, LAYER)
    assert at_layer == pytest.approx(LAYER * HIDDEN / vector.norm().item())


def test_projection_refuses_an_empty_answer_span():
    states = [torch.zeros(1, 3, HIDDEN) for _ in range(LAYER + 1)]
    with pytest.raises(steer.SteeringError, match="no tokens"):
        steer.response_avg_projection(HiddenStateModel(states), CharTokenizer(),
                                      "abc", "", torch.ones(HIDDEN), LAYER)


def test_span_stability_reports_a_merging_seam():
    assert steer.projection_span_is_stable(CharTokenizer(), "abc!", "de")
    assert not steer.projection_span_is_stable(MergingTokenizer(), "abc!", "de")


# --- a real generate(), with a KV cache and real decode steps ----------------------
# The tests above drive one block by hand. These drive `model.generate()` on a
# real Qwen2 of a few thousand parameters, so the prompt forward and every
# cached width-1 decode step run the same way they do at 7B. That is the property
# the GPU run pinned and nothing offline did.

QWEN_HIDDEN = 32
QWEN_PROMPT = [[1, 2, 3, 4, 5]]


def tiny_qwen():
    """A real `Qwen2ForCausalLM`, small enough to run in milliseconds on the CPU."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=QWEN_HIDDEN, intermediate_size=64,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)
    torch.manual_seed(3)
    return transformers.Qwen2ForCausalLM(config).eval()


@pytest.fixture
def qwen_vector_file(tmp_path):
    torch.manual_seed(4)
    path = tmp_path / "qwen_response_avg_diff.pt"
    torch.save(torch.randn(VECTOR_ROWS, QWEN_HIDDEN), path)
    return path


def qwen_generate(model, steering=None, new_tokens=8):
    ids = torch.tensor(QWEN_PROMPT)
    if steering is None:
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=new_tokens, do_sample=False,
                                  use_cache=True)
    with steer.ActivationSteering(model, steering), torch.no_grad():
        return model.generate(ids, max_new_tokens=new_tokens, do_sample=False,
                              use_cache=True)


def test_a_real_generate_is_byte_identical_at_coeff_zero(qwen_vector_file):
    """The GPU gate, offline: greedy decode with a cache, token ids compared."""
    model = tiny_qwen()
    baseline = qwen_generate(model)
    assert torch.equal(qwen_generate(model), baseline), "generate is not reproducible"

    at_zero = qwen_generate(model, steer.Steering(str(qwen_vector_file), LAYER, 0.0))
    assert torch.equal(at_zero, baseline)

    # controls, in the same test: without these a hook that never installed passes
    moved = qwen_generate(model, steer.Steering(str(qwen_vector_file), LAYER, 30.0))
    assert not torch.equal(moved, baseline), "the hook did not reach generate()"


def test_the_hook_fires_on_the_prompt_and_on_every_decode_step(qwen_vector_file):
    """Where `positions` is decided: one wide forward, then one per new token."""
    model = tiny_qwen()
    widths = []
    observer = model.model.layers[LAYER - 1].register_forward_hook(
        lambda _m, _i, out: widths.append(
            (out[0] if isinstance(out, tuple) else out).shape[1]))
    try:
        qwen_generate(model, steer.Steering(str(qwen_vector_file), LAYER, 1.0),
                      new_tokens=6)
    finally:
        observer.remove()
    assert widths == [len(QWEN_PROMPT[0])] + [1] * 5


def test_steering_a_real_generate_leaves_no_hook_behind(qwen_vector_file):
    model = tiny_qwen()
    qwen_generate(model, steer.Steering(str(qwen_vector_file), LAYER, 1.0))
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_the_engine_reads_the_vector_file_once_per_engine(qwen_vector_file, monkeypatch):
    """A reread per batch would let a mid-run file swap in under a stale sha."""
    model = tiny_qwen()
    steering = steer.Steering(str(qwen_vector_file), LAYER, 1.0)
    reads = []
    original = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (reads.append(a[0]),
                                                        original(*a, **k))[1])
    engine = steer.SteeredEngine(ProbeEngine(model.model.layers[LAYER - 1]), model,
                                 steering)
    for _batch in range(3):
        engine.generate(["a", "b"], object(), 0)
    assert len(reads) == 1


# --- the module must not drag upstream's import side effects in --------------------

def test_this_module_never_imports_upstreams_steerer():
    """`activation_steer` sets HF_HOME=hf_cache at import time.

    The tests above import it deliberately, in a process that loads no weights. A
    run must not: it would repoint the model cache from under the engine.
    """
    source = pathlib.Path(steer.__file__).read_text(encoding="utf-8")
    assert "import activation_steer" not in source
    assert "hf_cache" not in source
