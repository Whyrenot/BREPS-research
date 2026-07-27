"""
probe_sam3_api.py
=================
DIAGNOSTIC ONLY -- discover facebookresearch/sam3's real API on the server,
so heatmaps/sam3_adapter.py can be written against the actual source instead
of guesses.

Everything the rest of this repo needs from a predictor backend is listed in
scripts/refine_box_iou_grad.py::refine_box_by_iou_grad's docstring. For SAM1
and SAM2/SAM-HQ2 those facts were established by reading the installed
source with inspect.getsource on the server (see session_handoff.txt section
2: "never guess from memory"). This script does the same for SAM3, in one
pass, and answers the four questions that decide whether the gradient method
is portable at all:

    Q1  What builds an image model, and what does it return?
        (sam3.model_builder.build_sam3_image_model? something else?)
    Q2  Is there a predictor/processor with set_image()+predict(box=...),
        i.e. the SAM2ImagePredictor-shaped surface the repo calls?
    Q3  Does the model expose a prompt-encoder / mask-decoder pair with an
        IoU-prediction output -- the differentiable box -> predicted_iou
        path refine_box_by_iou_grad ascends?
    Q4  Does autograd actually flow from the predicted IoU back to the box
        coordinates (finite, non-zero d(pred_iou)/d(box))?

Q4 is the go/no-go: if the box prompt is not differentiable end-to-end, the
whole gradient defence cannot be ported to SAM3 and only the best_of_n /
stability baselines can run.

Nothing here imports the rest of the repo except env_dispatch, so it runs in
the bare `sam3` conda env.

Usage (from any env -- it re-execs itself into `sam3`):
    python scripts/probe_sam3_api.py \
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM3/sam3.pt \
        --image /.../user_study/FOR_TEST/images/100080.png \
        --out results/sam3_api_probe.txt

--checkpoint_path and --image are both optional: without a checkpoint it
still dumps the package layout / class signatures (Q1-Q3 statically), it
just cannot run the live smoke test (Q4).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# output plumbing: everything printed also lands in --out
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"== {title}")
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def safe(label: str, fn, *, limit_tb: int = 4):
    """Run fn(), print+swallow any exception. Returns (ok, value).

    Negative limit = keep the INNERMOST frames. A positive limit keeps the
    outermost ones, which for a failure deep inside a library shows only the
    call wrappers -- exactly what hid `self.image_encoder is None` behind
    torch's @no_grad decorator frame on the first real SAM3 run.
    """
    try:
        return True, fn()
    except Exception as e:  # noqa: BLE001 -- a probe must survive everything
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        for line in traceback.format_exc(limit=-limit_tb).splitlines():
            print(f"         {line}")
        return False, None


def show_signature(label: str, obj) -> None:
    ok, sig = safe(f"signature({label})", lambda: inspect.signature(obj))
    if ok:
        print(f"  {label}{sig}")


def show_source(label: str, obj, max_lines: int = 90) -> None:
    """Print the real source -- the whole point of this probe."""
    if obj is None:
        print(f"\n--- source: {label} --- (absent)")
        return
    sub(f"source: {label}")
    ok, src = safe(f"getsource({label})", lambda: inspect.getsource(obj))
    if not ok:
        return
    lines = src.splitlines()
    for line in lines[:max_lines]:
        print(f"  | {line}")
    if len(lines) > max_lines:
        print(f"  | ... ({len(lines) - max_lines} more lines, "
              f"file: {inspect.getsourcefile(obj)})")


# ---------------------------------------------------------------------------
# Q1/Q2: static package survey
# ---------------------------------------------------------------------------

_PREDICTOR_METHODS = ("set_image", "predict", "set_image_batch", "set_text_prompt",
                      "set_boxes", "__call__")
# class-name fragments worth reporting when walking the built model's tree
_COMPONENT_HINTS = ("promptencoder", "maskdecoder", "tracker", "sam2", "sammask",
                    "samprompt", "iou", "memory", "detector", "presence")


def survey_package(pkg_name: str = "sam3") -> "object | None":
    section(f"Q1  package layout: {pkg_name}")
    try:
        pkg = importlib.import_module(pkg_name)
    except ModuleNotFoundError as e:
        print(f"  [FAIL] import {pkg_name}: {e}")
        if e.name and e.name.split(".")[0] != pkg_name:
            # `sam3` IS installed, one of ITS imports is what failed. Do not
            # send the reader off to re-create the env for a missing leaf dep.
            print(f"\n  -> `{pkg_name}` is installed, but importing it needs "
                  f"`{e.name}`, which is absent.\n"
                  f"     sam3 does not declare every dependency it imports. Fix:\n"
                  f"         conda run -n sam3 pip install {e.name}\n"
                  f"     and add it to setup_sam3_env() in scripts/setup_repo.sh "
                  f"so the next\n     env build does not hit this again.")
        else:
            print(f"\n  -> `{pkg_name}` is not installed in this interpreter "
                  f"({sys.executable}).\n"
                  f"     Create the env first: ./scripts/setup_repo.sh --envs sam3")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] import {pkg_name}: {type(e).__name__}: {e}")
        for line in traceback.format_exc(limit=4).splitlines()[-5:]:
            print(f"         {line}")
        return None

    print(f"  {pkg_name}.__file__    = {getattr(pkg, '__file__', None)}")
    print(f"  {pkg_name}.__version__ = {getattr(pkg, '__version__', '<none>')}")
    print(f"  {pkg_name}.__path__    = {list(getattr(pkg, '__path__', []))}")
    print(f"  dir({pkg_name})        = {[n for n in dir(pkg) if not n.startswith('_')]}")

    sub("submodules (pkgutil.walk_packages, not imported)")
    ok, mods = safe("walk_packages", lambda: sorted(
        m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}.")))
    if ok:
        for name in mods:
            print(f"  {name}")
    return pkg


def survey_builders(pkg_name: str = "sam3") -> None:
    """Every public callable whose name starts with build_/create_/load_ or
    that looks like a predictor/processor class, across sam3.*."""
    section("Q1  builders and predictor/processor candidates")
    ok, pkg = safe(f"import {pkg_name}", lambda: importlib.import_module(pkg_name))
    if not ok:
        return

    module_names = [pkg_name]
    ok, walked = safe("walk_packages", lambda: [
        m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}.")])
    if ok:
        # skip obviously heavy / optional subtrees to keep import errors rare
        module_names += [m for m in walked if ".test" not in m and "benchmark" not in m]

    builders: list[tuple[str, object]] = []
    predictors: list[tuple[str, type]] = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {mod_name}: {type(e).__name__}: {e}")
            continue
        for attr, obj in vars(mod).items():
            if attr.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != mod_name:
                continue  # re-export, report it where it is defined
            if inspect.isfunction(obj) and attr.split("_")[0] in ("build", "create", "load"):
                builders.append((f"{mod_name}.{attr}", obj))
            elif inspect.isclass(obj):
                have = [m for m in _PREDICTOR_METHODS if hasattr(obj, m)]
                if "set_image" in have or ("predict" in have and "__call__" in have):
                    predictors.append((f"{mod_name}.{attr}", obj))

    sub("builder functions")
    for name, fn in builders:
        show_signature(name, fn)
        doc = (inspect.getdoc(fn) or "").splitlines()
        if doc:
            print(f"      \"{doc[0][:110]}\"")
    if not builders:
        print("  (none found -- check the submodule list above by hand)")

    sub("predictor / processor classes (have set_image, or predict+__call__)")
    for name, cls in predictors:
        print(f"  {name}   mro={[c.__name__ for c in cls.__mro__[:4]]}")
        for meth in _PREDICTOR_METHODS:
            if hasattr(cls, meth):
                show_signature(f"    .{meth}", getattr(cls, meth))
    if not predictors:
        print("  (none found -- SAM3 may expose only a functional API)")

    # The three methods the repo actually calls, in full.
    for name, cls in predictors:
        for meth in ("set_image", "predict"):
            if hasattr(cls, meth):
                show_source(f"{name}.{meth}", getattr(cls, meth))


# ---------------------------------------------------------------------------
# Q3: the built model's module tree
# ---------------------------------------------------------------------------

def _build_untrained(device: str):
    """Architecture-only fallback: SAM3's interactive predictor with random
    weights. Returns (model, predictor) or (None, None)."""
    sub("building the interactive predictor WITHOUT weights (build_tracker)")
    try:
        from heatmaps.sam3_adapter import build_sam3_interactive_untrained
        pred = build_sam3_interactive_untrained(device=device)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {type(e).__name__}: {e}")
        for line in traceback.format_exc(limit=4).splitlines()[-6:]:
            print(f"         {line}")
        return None, None
    print(f"  [OK] {type(pred).__module__}.{type(pred).__name__} "
          f"over {type(pred.model).__name__}")
    print("  !! RANDOM WEIGHTS. Q3 (contract) and Q4 (autograd) are properties\n"
          "     of the architecture and stay valid; every NUMBER below is noise.\n"
          "     Nothing built this way may reach results/.")
    return pred.model, pred


def build_model(builder_path: str | None, checkpoint: str | None, device: str,
                untrained: bool = False):
    """Build SAM3's image model.

    Primary path goes through heatmaps.sam3_adapter.build_sam3_model, which
    encodes the two settings this repo needs and which are NOT the builder's
    defaults (enable_inst_interactivity=True, and an explicit bpe_path around
    sam3 0.1.4's broken default) -- so probing exercises the same call
    load_sam3_model makes, rather than a lookalike that could diverge.
    Falls back to raw signature-adaptive calls if that import fails.

    Returns (model, native_predictor_or_None) -- the untrained path builds the
    predictor directly, so it has no image model to re-derive one from.
    """
    section("Q1  build the image model")
    import torch  # noqa: F401 -- confirms torch is importable in this env

    if untrained:
        print("  --untrained requested: skipping the checkpoint entirely.")
        return _build_untrained(device)

    if builder_path is None:
        sub("via heatmaps.sam3_adapter.build_sam3_model (what load_sam3_model uses)")
        try:
            from heatmaps.sam3_adapter import build_sam3_model, find_bpe_path
            print(f"  bpe_path = {find_bpe_path()}")
            model = build_sam3_model(checkpoint=checkpoint, device=device,
                                     enable_inst_interactivity=True)
            if model is not None:
                print(f"  [OK] built: {type(model).__name__}")
                return model, None
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {type(e).__name__}: {e}")
            for line in traceback.format_exc(limit=4).splitlines()[-6:]:
                print(f"         {line}")
            try:
                from heatmaps.sam3_adapter import _is_gated_repo_error
                gated = _is_gated_repo_error(e)
            except Exception:  # noqa: BLE001
                gated = False
            if gated:
                # Every other builder spelling downloads from the same gated
                # repo and fails identically. The architecture-only path does
                # not need weights at all, and Q3/Q4 do not depend on them.
                print("  -> BLOCKED on Hugging Face access, not on the API.")
                return _build_untrained(device)
            print("  -> falling back to raw builder calls below")

    candidates = [builder_path] if builder_path else [
        "sam3.model_builder.build_sam3_image_model",
    ]
    for path in candidates:
        mod_name, _, fn_name = path.rpartition(".")
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {path}: {type(e).__name__}: {e}")
            continue

        show_signature(path, fn)
        show_source(path, fn, max_lines=60)
        params = inspect.signature(fn).parameters

        # sam3 0.1.4's bpe_path default resolves outside the installed
        # package; supply it here too, or this fallback fails on that instead
        # of on whatever we are actually trying to learn about.
        base_kwargs: dict = {}
        if "bpe_path" in params:
            try:
                from heatmaps.sam3_adapter import find_bpe_path
                base_kwargs["bpe_path"] = find_bpe_path()
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] no bpe_path: {e}")
        if "enable_inst_interactivity" in params:
            base_kwargs["enable_inst_interactivity"] = True

        # Try the kwarg spellings a checkpoint could plausibly have, then no-arg
        # (SAM3 may pull the gated checkpoint straight from Hugging Face).
        attempts: list[dict] = []
        if checkpoint:
            for key in ("checkpoint_path", "ckpt_path", "checkpoint", "ckpt", "model_path"):
                if key in params:
                    attempts.append({key: checkpoint})
            if not attempts:
                attempts.append({"__positional__": checkpoint})
        attempts.append({})

        for kwargs in attempts:
            pos = kwargs.pop("__positional__", None)
            label = f"{fn_name}({pos or ''}{', ' if pos and kwargs else ''}" \
                    f"{', '.join(f'{k}=...' for k in kwargs)})"
            kwargs = {**base_kwargs, **kwargs}
            if "device" in params:
                kwargs.setdefault("device", device)
            ok, model = safe(label, lambda: fn(pos, **kwargs) if pos else fn(**kwargs))
            if ok and model is not None:
                print(f"  [OK] built via {label}")
                return model, None
    print("  -> no builder worked; trying the architecture-only path so Q3/Q4\n"
          "     are still answered.")
    return _build_untrained(device)


def dump_module_tree(model, max_depth: int = 2) -> None:
    section("Q3  model structure")
    print(f"  type(model) = {type(model).__module__}.{type(model).__name__}")
    print(f"  mro         = {[c.__name__ for c in type(model).__mro__[:6]]}")
    print(f"  source file = {inspect.getsourcefile(type(model))}")

    sub("public non-module attributes (config-ish: image size, thresholds, ...)")
    for attr in sorted(a for a in dir(model) if not a.startswith("_")):
        ok, val = safe(f"getattr({attr})", lambda a=attr: getattr(model, a))
        if not ok:
            continue
        if isinstance(val, (int, float, bool, str, tuple, list)) and not callable(val):
            print(f"  {attr:36s} = {str(val)[:90]}")

    sub(f"named_children (depth<={max_depth})")
    ok, _ = safe("named_modules", lambda: None)
    for name, child in model.named_modules():
        depth = name.count(".") + 1 if name else 0
        if depth > max_depth:
            continue
        print(f"  {'  ' * depth}{name or '<root>':44s} {type(child).__name__}")

    sub("modules whose class name looks like a prompt-encoder / mask-decoder")
    hits = []
    for name, child in model.named_modules():
        cname = type(child).__name__.lower()
        if any(h in cname for h in _COMPONENT_HINTS):
            hits.append((name, child))
    for name, child in hits:
        print(f"  {name:52s} {type(child).__module__}.{type(child).__name__}")
    if not hits:
        print("  (no name matches -- inspect the full tree above by hand)")

    # The two forwards the gradient path calls, plus get_dense_pe.
    seen: set[int] = set()
    for name, child in hits:
        cname = type(child).__name__.lower()
        if not ("promptencoder" in cname or "maskdecoder" in cname):
            continue
        if id(type(child)) in seen:
            continue
        seen.add(id(type(child)))
        show_signature(f"{name}.forward", type(child).forward)
        show_source(f"{name}.forward", type(child).forward, max_lines=110)
        if hasattr(child, "get_dense_pe"):
            show_signature(f"{name}.get_dense_pe", type(child).get_dense_pe)


def deep_dive_interactive(model, predictor) -> None:
    """Q2b -- how SAM3 wires its interactive predictor to the vision backbone.

    Raised by the first real run: the predictor IS attached to the model, but
    `predictor.model` is None, so its set_image() dies on
    `self.model.forward_image(...)`. SAM3 evidently shares ONE vision backbone
    across the concept path and the box path -- Sam3Processor.set_image()
    computes `backbone_out = self.model.backbone.forward_image(image)` and
    then reaches into `inst_interactive_predictor.model.sam_mask_decoder`, so
    the features are meant to be injected, not recomputed.

    This dumps everything needed to decide how to drive that: the predictor's
    __init__, the FULL builder (its tail was truncated before and is what
    actually wires inst_predictor), the image model's own structure, and
    Sam3Processor's inference path.
    """
    section("Q2b  how the interactive predictor gets its image features")

    inner = getattr(predictor, "model", "<no attribute>")
    print(f"  predictor.model = {type(inner).__name__ if inner is not None else 'None'}")
    if inner is None:
        print("  -> THIS is why set_image() fails: SAM3InteractiveImagePredictor."
              "set_image()\n"
              "     calls self.model.forward_image(), and self.model is None. The "
              "predictor\n"
              "     cannot encode an image by itself; the features have to come "
              "from the\n"
              "     shared backbone. Read the builder tail + Sam3Processor below "
              "to see how.")

    sub("predictor's own state")
    for attr in ("_bb_feat_sizes", "mask_threshold", "device", "_is_image_set",
                 "_orig_hw", "_features"):
        if hasattr(predictor, attr):
            val = getattr(predictor, attr)
            print(f"  {attr:20s} = {str(val)[:100] if not hasattr(val, 'shape') else val.shape}")

    show_source("SAM3InteractiveImagePredictor.__init__", type(predictor).__init__)
    show_source("SAM3InteractiveImagePredictor._predict",
                getattr(type(predictor), "_predict", None), max_lines=120)
    show_source("SAM3InteractiveImagePredictor._prep_prompts",
                getattr(type(predictor), "_prep_prompts", None), max_lines=60)

    # The tail of this is what actually attaches inst_predictor -- it was cut
    # off at 60 lines in the Q1 dump, and it is the missing piece.
    sub("the FULL builder (its tail wires inst_predictor and loads the checkpoint)")
    try:
        from sam3.model_builder import build_sam3_image_model
        show_source("build_sam3_image_model", build_sam3_image_model, max_lines=400)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {type(e).__name__}: {e}")

    sub("the image model that owns the shared backbone")
    print(f"  type(model) = {type(model).__module__}.{type(model).__name__}")
    for name, child in model.named_children():
        print(f"    {name:34s} {type(child).__name__}")
    for attr in ("backbone", "inst_interactive_predictor"):
        got = getattr(model, attr, None)
        print(f"  model.{attr} = {type(got).__name__ if got is not None else None}")
    if getattr(model, "backbone", None) is not None:
        show_signature("  model.backbone.forward_image",
                       type(model.backbone).forward_image)
    show_source(f"{type(model).__name__}.__init__", type(model).__init__, max_lines=120)

    sub("Sam3Processor: the supported way to run this model")
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        show_signature("Sam3Processor.__init__", Sam3Processor.__init__)
        show_source("Sam3Processor.__init__", Sam3Processor.__init__, max_lines=60)
        for meth in sorted(m for m in dir(Sam3Processor) if not m.startswith("__")):
            fn = getattr(Sam3Processor, meth, None)
            if callable(fn) and meth not in ("set_image",):
                show_signature(f"  .{meth}", fn)
        for meth in ("predict", "inference", "set_boxes", "set_visual_prompt",
                     "set_geometry_prompt", "predict_boxes"):
            fn = getattr(Sam3Processor, meth, None)
            if fn is not None:
                show_source(f"Sam3Processor.{meth}", fn, max_lines=90)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {type(e).__name__}: {e}")


def check_sam2_surface(model, predictor=None) -> None:
    """Does SAM3 happen to expose the exact attribute names the existing
    SAM2 branch of refine_box_by_iou_grad uses? If it does, the port is a
    no-op; if not, this lists precisely what heatmaps/sam3_adapter.py has to
    synthesise."""
    section("Q3  SAM2-branch compatibility checklist")
    model_attrs = ["sam_prompt_encoder", "sam_mask_decoder", "prompt_encoder",
                   "mask_decoder", "image_encoder", "mask_threshold", "image_size"]
    pred_attrs = ["predict_torch", "predict", "set_image", "_features", "_orig_hw",
                  "original_size", "input_size", "_transforms", "mask_threshold",
                  "model", "interm_features"]

    sub("predictor.model (what refine_box_by_iou_grad reads off `model`)")
    for a in model_attrs:
        mark = "OK " if hasattr(model, a) else "-- "
        print(f"  [{mark}] model.{a}")

    sub("predictor itself")
    if predictor is None:
        print("  (no predictor instance built -- rerun with --checkpoint_path)")
        return
    for a in pred_attrs:
        mark = "OK " if hasattr(predictor, a) else "-- "
        print(f"  [{mark}] predictor.{a}")
    print("\n  NOTE: refine_box_by_iou_grad picks its backend branch with")
    print("        is_sam2 = not hasattr(predictor, 'predict_torch')")
    print("        is_samhq = hasattr(predictor, 'interm_features')")

    tr = getattr(predictor, "_transforms", None)
    if tr is not None:
        sub("predictor._transforms (box scaling + postprocess_masks)")
        print(f"  type = {type(tr).__module__}.{type(tr).__name__}")
        for a in ("resolution", "transform_boxes", "transform_coords", "postprocess_masks"):
            if hasattr(tr, a):
                val = getattr(tr, a)
                if callable(val):
                    show_signature(f"  .{a}", val)
                else:
                    print(f"  .{a} = {val}")
            else:
                print(f"  [--] .{a}")


# ---------------------------------------------------------------------------
# Q4: live smoke test + the autograd go/no-go
# ---------------------------------------------------------------------------

def _demo_image(image_path: str | None):
    import numpy as np
    if image_path:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # synthetic fallback: a bright rectangle on noise, so a box prompt has
    # something real to segment even without the user_study data mounted.
    rng = np.random.default_rng(0)
    img = (rng.integers(0, 60, size=(480, 640, 3))).astype("uint8")
    img[150:330, 200:440] = 220
    return img


def live_smoke(model, predictor, image_path: str | None, device: str) -> None:
    section("Q2  live smoke test: set_image + box prompt")
    import numpy as np
    import torch

    img = _demo_image(image_path)
    H, W = img.shape[:2]
    print(f"  image {W}x{H} from {image_path or '<synthetic 640x480 rectangle>'}")
    # a box around the bright rectangle (or the given image's centre region)
    box = np.array([W * 0.30, H * 0.30, W * 0.70, H * 0.70], dtype=np.float32)
    print(f"  box (original px) = {box.tolist()}")

    if predictor is None:
        print("  (no predictor -- skipping)")
        return

    ok, _ = safe("predictor.set_image(img)", lambda: predictor.set_image(img))
    if not ok:
        return

    sub("cached state after set_image (the image embedding the box is decoded against)")
    for a in ("_features", "_orig_hw", "original_size", "input_size", "_is_image_set"):
        if not hasattr(predictor, a):
            continue
        val = getattr(predictor, a)
        if isinstance(val, dict):
            print(f"  {a}:")
            for k, v in val.items():
                if torch.is_tensor(v):
                    print(f"    {k:24s} tensor{tuple(v.shape)} {v.dtype} {v.device}")
                elif isinstance(v, (list, tuple)):
                    shapes = [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in v]
                    print(f"    {k:24s} list(len={len(v)}) shapes={shapes}")
                else:
                    print(f"    {k:24s} {type(v).__name__}")
        elif torch.is_tensor(val):
            print(f"  {a}: tensor{tuple(val.shape)}")
        else:
            print(f"  {a}: {val}")

    sub("predictor.predict(box=...) -- both multimask settings")
    for mm in (False, True):
        ok, out = safe(f"predict(multimask_output={mm})", lambda mm=mm: predictor.predict(
            point_coords=None, point_labels=None, box=box, multimask_output=mm))
        if not ok:
            continue
        print(f"  multimask_output={mm}: returned {len(out)} values")
        for i, v in enumerate(out):
            if hasattr(v, "shape"):
                print(f"    [{i}] {type(v).__name__} shape={tuple(v.shape)} "
                      f"dtype={getattr(v, 'dtype', '?')} "
                      f"range=[{float(np.min(np.asarray(v))):.3f}, "
                      f"{float(np.max(np.asarray(v))):.3f}]")
            else:
                print(f"    [{i}] {type(v).__name__} = {v}")

    autograd_check(model, predictor, device)


def autograd_check(model, predictor, device: str) -> None:
    """THE go/no-go: is d(predicted_iou)/d(box) finite and non-zero?

    Mirrors refine_box_by_iou_grad's inner loop as closely as possible without
    knowing SAM3's exact names: locate a prompt encoder + mask decoder, feed a
    box that requires grad, backward through the IoU-prediction output.
    """
    section("Q4  autograd through the box prompt (GO / NO-GO)")
    import torch

    def find(kind: str):
        for name, child in model.named_modules():
            if kind in type(child).__name__.lower():
                return name, child
        return None, None

    pe_name, prompt_encoder = find("promptencoder")
    md_name, mask_decoder = find("maskdecoder")
    print(f"  prompt encoder : {pe_name or '<NOT FOUND>'} "
          f"({type(prompt_encoder).__name__ if prompt_encoder else '-'})")
    print(f"  mask decoder   : {md_name or '<NOT FOUND>'} "
          f"({type(mask_decoder).__name__ if mask_decoder else '-'})")
    if prompt_encoder is None or mask_decoder is None:
        print("  -> NO-GO via this route. SAM3 has no SAM-style prompt/mask module\n"
              "     pair under those class names. Read the module tree in Q3 and\n"
              "     look for the DETR-style decoder instead; the gradient defence\n"
              "     then has to ascend whatever confidence/IoU output that has.")
        return

    feats = getattr(predictor, "_features", None) if predictor is not None else None
    if not isinstance(feats, dict) or "image_embed" not in feats:
        print("  -> cannot run: no cached {'image_embed': ...} from set_image().\n"
              "     Check the Q2 dump above for what set_image() actually caches\n"
              "     and re-run this section against those names.")
        return

    for p in model.parameters():
        p.requires_grad_(False)

    img_embed = feats["image_embed"][-1].unsqueeze(0)
    high_res = [f[-1].unsqueeze(0) for f in feats.get("high_res_feats", [])]
    res = float(getattr(getattr(predictor, "_transforms", None), "resolution", 1024))
    # box in the model's own input frame, as SAM2's transform_boxes(normalize=True)
    # would produce; the exact frame does not matter for a gradient-existence test.
    params = torch.tensor([0.5 * res, 0.5 * res, 0.4 * res, 0.4 * res],
                          device=device, requires_grad=True)

    def to_box(p):
        return torch.stack([p[0] - p[2] / 2, p[1] - p[3] / 2,
                            p[0] + p[2] / 2, p[1] + p[3] / 2])

    pe_sig = inspect.signature(type(prompt_encoder).forward).parameters
    md_sig = inspect.signature(type(mask_decoder).forward).parameters
    print(f"  prompt_encoder.forward params = {list(pe_sig)[1:]}")
    print(f"  mask_decoder.forward   params = {list(md_sig)[1:]}")

    def encode(box, style: str):
        if style == "sam1":     # boxes= kwarg (SAM1 / SAM-HQ)
            return prompt_encoder(points=None, boxes=box[None, :], masks=None)
        # SAM2: box as two corner points labelled 2 (top-left) / 3 (bottom-right)
        coords = box.reshape(1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=box.device)
        return prompt_encoder(points=(coords, labels), boxes=None, masks=None)

    styles = ["sam2", "sam1"] if "boxes" in pe_sig else ["sam2"]
    for style in styles:
        sub(f"attempt: box encoded {style}-style")
        try:
            with torch.enable_grad():
                box = to_box(params)
                sparse, dense = encode(box, style)
                kwargs = dict(
                    image_embeddings=img_embed,
                    image_pe=prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=True,
                )
                if "repeat_image" in md_sig:
                    kwargs["repeat_image"] = False
                if "high_res_features" in md_sig and high_res:
                    kwargs["high_res_features"] = high_res
                if "hq_token_only" in md_sig:
                    kwargs["hq_token_only"] = False
                out = mask_decoder(**kwargs)
                print(f"  decoder returned {len(out)} values: "
                      f"{[tuple(o.shape) if torch.is_tensor(o) else type(o).__name__ for o in out]}")
                low_res, iou_pred = out[0], out[1]
                print(f"  iou_pred = {iou_pred.detach().flatten().tolist()}")
                score = iou_pred.flatten().max()
                score.backward()
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {type(e).__name__}: {e}")
            print("  " + traceback.format_exc(limit=4).replace("\n", "\n  "))
            continue

        g = params.grad
        if g is None:
            print("  -> NO-GO: box.grad is None (graph is broken somewhere, most\n"
                  "     likely a .detach()/no_grad inside the prompt encoder).")
            return
        finite = bool(torch.isfinite(g).all())
        nonzero = float(g.abs().max())
        print(f"  d(pred_iou)/d(cx,cy,w,h) = {g.tolist()}")
        print(f"  finite={finite}  max|grad|={nonzero:.3e}")
        if finite and nonzero > 0:
            print(f"\n  ==> GO. Gradient ascent on SAM3's predicted IoU is possible;\n"
                  f"      encode boxes {style}-style, decoder kwargs "
                  f"{sorted(k for k in kwargs if k != 'image_pe')}.")
        else:
            print("\n  ==> NO-GO: gradient is zero or non-finite. If it is exactly\n"
                  "      zero, the box likely goes through a rounding/argmax step;\n"
                  "      if NaN/inf, check the dtype (autocast/bf16) of the cached\n"
                  "      image embedding.")
        return


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", default=None,
                   help="SAM3 checkpoint. Omit to survey the API statically "
                        "(no model built, no Q4 autograd verdict).")
    p.add_argument("--builder", default=None,
                   help="fully-qualified builder, e.g. "
                        "sam3.model_builder.build_sam3_image_model (default: "
                        "try the known spellings in order)")
    p.add_argument("--predictor_class", default=None,
                   help="fully-qualified predictor/processor class to wrap the "
                        "model in, e.g. sam3.processor.Sam3Processor (default: "
                        "the first class found with set_image + predict)")
    p.add_argument("--image", default=None,
                   help="image for the live smoke test (default: synthetic)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--untrained", action="store_true", default=False,
                   help="build the interactive predictor with RANDOM weights "
                        "via build_tracker(), skipping the checkpoint, Hugging "
                        "Face and the text encoder entirely. Q3 (contract) and "
                        "Q4 (autograd) are properties of the architecture, so "
                        "they are still answered; no number is meaningful. "
                        "Used automatically when the gated repo refuses.")
    p.add_argument("--out", default="results/sam3_api_probe.txt")
    p.add_argument("--no_dispatch", action="store_true",
                   help="do not re-exec into the `sam3` conda env (use when "
                        "already inside it, or to probe another env)")
    return p.parse_args()


def _instantiate_predictor(model, class_path: str | None):
    """Get SAM3's own predictor.

    Preferred: the SAM3InteractiveImagePredictor the builder already attached
    to the model when enable_inst_interactivity=True -- constructing a fresh
    one from the image model would be wrong anyway (it wraps the TRACKER, not
    the image model). Only if that is absent do we scan for classes with
    set_image and try instantiating them.
    """
    section("Q2  instantiate the predictor / processor")
    if class_path is None:
        try:
            from heatmaps.sam3_adapter import get_interactive_predictor
            attached = get_interactive_predictor(model)
            if attached is not None:
                print(f"  [OK] model already carries "
                      f"{type(attached).__module__}.{type(attached).__name__}")
                for meth in ("set_image", "predict", "predict_torch", "_predict"):
                    if hasattr(attached, meth):
                        show_signature(f"  .{meth}", getattr(attached, meth))
                return attached
            print("  model carries no interactive predictor "
                  "(built without enable_inst_interactivity?)")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] get_interactive_predictor: {type(e).__name__}: {e}")

    candidates: list[str] = [class_path] if class_path else []
    if not candidates:
        try:
            pkg = importlib.import_module("sam3")
            for m in pkgutil.walk_packages(pkg.__path__, prefix="sam3."):
                try:
                    mod = importlib.import_module(m.name)
                except Exception:  # noqa: BLE001
                    continue
                for attr, obj in vars(mod).items():
                    if (inspect.isclass(obj) and hasattr(obj, "set_image")
                            and getattr(obj, "__module__", None) == m.name):
                        candidates.append(f"{m.name}.{attr}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] scanning for predictor classes: {e}")

    print(f"  candidates: {candidates or '<none>'}")
    for path in candidates:
        mod_name, _, cls_name = path.rpartition(".")
        ok, cls = safe(f"import {path}",
                       lambda: getattr(importlib.import_module(mod_name), cls_name))
        if not ok:
            continue
        show_signature(f"{path}.__init__", cls.__init__)
        ok, pred = safe(f"{cls_name}(model)", lambda: cls(model))
        if ok and pred is not None:
            print(f"  [OK] predictor = {path}")
            return pred
    print("  -> no predictor could be instantiated with (model); check the\n"
          "     __init__ signatures printed above (it may need a config, a\n"
          "     checkpoint path, or a from_pretrained() classmethod).")
    return None


def main():
    args = parse_args()
    if not args.no_dispatch:
        from heatmaps.env_dispatch import maybe_dispatch_to_env
        maybe_dispatch_to_env("SAM3", __file__)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, f)

    try:
        section("environment")
        print(f"  executable = {sys.executable}")
        print(f"  python     = {sys.version.split()[0]}")
        ok, torch = safe("import torch", lambda: importlib.import_module("torch"))
        if ok:
            print(f"  torch      = {torch.__version__}")
            print(f"  cuda       = {torch.version.cuda}  available={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  device 0   = {torch.cuda.get_device_name(0)} "
                      f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
        device = f"cuda:{args.gpu}" if (ok and torch.cuda.is_available()) else "cpu"
        print(f"  device     = {device}")

        survey_package("sam3")
        survey_builders("sam3")

        model, native = build_model(args.builder, args.checkpoint_path, device,
                                    untrained=args.untrained)
        if model is None:
            print("\n[probe] stopping after the static survey (no model built).")
            return

        safe("model.eval()", lambda: model.eval())
        dump_module_tree(model)
        if native is not None:
            section("Q2  instantiate the predictor / processor")
            print(f"  using the predictor built alongside the model: "
                  f"{type(native).__module__}.{type(native).__name__}")
            predictor = native
        else:
            predictor = _instantiate_predictor(model, args.predictor_class)
        check_sam2_surface(model, predictor)
        if predictor is not None:
            deep_dive_interactive(model, predictor)
            # Adopt exactly as load_sam3_model does, so the smoke test below
            # exercises the real path (this is also what installs a working
            # set_image over the shared backbone).
            section("Q2c  adopt_native_predictor (what load_sam3_model returns)")
            ok, adopted = safe(
                "adopt_native_predictor(predictor, image_model=model)",
                lambda: __import__("heatmaps.sam3_adapter", fromlist=["x"])
                .adopt_native_predictor(predictor, image_model=model, device=device))
            if ok and adopted is not None:
                predictor = adopted
                patched = "set_image" in vars(predictor)
                print(f"  [OK] adopted; set_image patched onto the instance: {patched}")
                if patched:
                    print("       (the tracker has no vision backbone of its own, "
                          "so features\n        come from Sam3Image.backbone -- see Q2b)")
        live_smoke(model, predictor, args.image, device)
    finally:
        sys.stdout = real_stdout
        f.close()
        print(f"\nSaved probe report -> {out_path}")


if __name__ == "__main__":
    main()
