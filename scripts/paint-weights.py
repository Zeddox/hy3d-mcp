"""Rewrite the turbo pipeline's two .bin weights as safetensors.

The turbo subfolder publishes its text_encoder and vae only as pickles, and
transformers >= 5 refuses torch.load outright under torch < 2.6 (CVE-2025-32434)
-- a blanket version gate, not a claim about these particular files. Converting
once takes the pickle out of the loading path for good.
"""
import sys, torch, json
from pathlib import Path
from safetensors.torch import save_file

ROOT = Path(sys.argv[1])
JOBS = [("text_encoder/pytorch_model.bin", "text_encoder/model.safetensors"),
        ("vae/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.safetensors")]

for src_rel, dst_rel in JOBS:
    src, dst = ROOT / src_rel, ROOT / dst_rel
    if dst.is_file():
        print("skip  %s (already converted)" % dst_rel)
        continue
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    # safetensors refuses aliased storage, which CLIP checkpoints often carry
    # (tied embeddings, and a position_ids buffer sharing a base tensor).
    sd = {k: v.contiguous().clone() for k, v in sd.items()}
    save_file(sd, dst, metadata={"format": "pt"})
    print("ok    %s -> %s  (%d tensors, %.2f GB)"
          % (src_rel, dst_rel, len(sd), dst.stat().st_size / 1e9))


# The turbo unet is the awkward one, and it goes the other way. Its loader is
# custom code shipped inside the checkpoint, and that code reads
# diffusion_pytorch_model.bin by name -- the 3.72GB safetensors beside it is
# never opened. Rather than pull the 7.33GB fp32 pickle from the hub, build
# the .bin from the safetensors we already have.
#
# The safetensors holds 1571 tensors; the model the loader constructs wants
# 1535 and loads strict. The 36 extras are the IP-adapter path -- 32
# attn2.to_k_ip / to_v_ip plus the four image_proj_model weights -- which this
# loader never builds, because it hardcodes is_turbo = False, and which
# nothing in the pipeline references. Dropping them reproduces exactly the
# state dict upstream's own .bin must contain, since a strict load of anything
# wider would fail.
def build_unet_bin(root: Path) -> None:
    import importlib.util
    from safetensors.torch import load_file

    dst = root / "unet/diffusion_pytorch_model.bin"
    if dst.is_file():
        print("skip  unet/diffusion_pytorch_model.bin (already built)")
        return
    src = root / "unet/diffusion_pytorch_model.safetensors"
    spec = importlib.util.spec_from_file_location("hy3d_turbo_unet", root / "unet/modules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.loads((root / "unet/config.json").read_text())
    want = set(mod.UNet2p5DConditionModel(mod.UNet2DConditionModel(**cfg)).state_dict())

    sd = load_file(src)
    missing = want - set(sd)
    if missing:
        raise SystemExit("the safetensors is missing %d tensors the loader "
                         "requires, e.g. %s" % (len(missing), sorted(missing)[:3]))
    dropped = sorted(set(sd) - want)
    stray = [k for k in dropped
             if not (k.endswith(("to_k_ip.weight", "to_v_ip.weight"))
                     or ".image_proj_model." in k)]
    if stray:
        raise SystemExit("unexpected extra tensors beyond the IP-adapter "
                         "layers: %s" % stray[:5])
    torch.save({k: sd[k] for k in want}, dst)
    print("ok    unet safetensors -> .bin  (%d tensors, %d ip-adapter dropped, "
          "%.2f GB)" % (len(want), len(dropped), dst.stat().st_size / 1e9))


build_unet_bin(ROOT)
