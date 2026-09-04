from pathlib import Path

import pytest


def test_upstream_specs_are_exactly_pinned():
    from oemseg.upstreams import get_upstream

    geosa = get_upstream("GeoSA-BaSA")
    assert geosa.repo == "https://github.com/mmmll23/GeoSA-BaSA.git"
    assert geosa.revision == "67cec152349db6b29bf14e991e80f01fd365f73c"
    assert geosa.dest.name == "GeoSA-BaSA"

    hg = get_upstream("hg-rsovsseg")
    assert hg.repo == "https://github.com/HuangWBill/HG-RSOVSSeg.git"
    assert hg.revision == "58e1df1f68f15920e02320dc54fe7441acf37770"

    repstdc = get_upstream("RepSTDC")
    assert repstdc.repo == "https://github.com/jayeliu/RepSTDC.git"
    assert repstdc.revision == "549c2e0cf81963aeae8e5c6b9c83a3aa496a8407"


def test_official_train_commands_select_published_entrypoints():
    from oemseg.upstreams import build_train_command

    assert build_train_command("geosa_basa")[1:] == [
        "tools/train.py",
        "configs/dinov2/geosa_basa_dinov2_mask2former_512x512_bs1x4_whumix.py",
    ]
    assert build_train_command("hg_rsovsseg")[1:] == [
        "tools/train/train.py",
        "--config",
        "configs/OpenEarthMap_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_openearthmap-512x512.py",
        "--work-dir",
        "result/HG-RSOVSSeg/OpenEarthMap/",
    ]
    assert build_train_command("repstdc")[1:] == [
        "tools/train.py",
        "config/repstdc/repstdc-ca_512x512_80k_oem.py",
    ]


def test_official_eval_commands_require_checkpoint_and_select_oem_paths():
    from oemseg.upstreams import build_eval_command

    with pytest.raises(ValueError, match="checkpoint"):
        build_eval_command("repstdc", None)

    assert build_eval_command("repstdc", Path("model.pth"))[1:] == [
        "tools/test.py",
        "config/repstdc/repstdc-ca_512x512_80k_oem.py",
        "model.pth",
    ]
    assert build_eval_command("hg_rsovsseg", Path("iter_80000.pth"))[1:] == [
        "tools/test.py",
        "--config",
        "configs/OpenEarthMap_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_openearthmap-512x512_test.py",
        "--checkpoint",
        "iter_80000.pth",
        "--work-dir",
        "result/HG-RSOVSSeg/OpenEarthMap/test/",
    ]
    geosa = build_eval_command("geosa_basa", Path("checkpoints/whumix.pth"))[1:]
    assert geosa[:3] == [
        "tools/test.py",
        "configs/dinov2/geosa_basa_dinov2_mask2former_512x512_bs1x4_whumix.py",
        "checkpoints/whumix.pth",
    ]
    assert geosa[-2:] == ["--backbone", "./pretrained_weights/dinov2_converted.pth"]


def test_passthrough_args_are_appended_and_invalid_model_is_rejected():
    from oemseg.upstreams import build_train_command, get_upstream

    assert build_train_command("repstdc", ["--work-dir", "runs/rep"])[-2:] == [
        "--work-dir",
        "runs/rep",
    ]
    with pytest.raises(ValueError, match="unknown paper model"):
        get_upstream("not-a-model")


def test_paper_models_cli_dry_run_does_not_require_checkout(capsys):
    from scripts import paper_models

    assert paper_models.main(["train", "repstdc", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "repstdc-ca_512x512_80k_oem.py" in out
    assert "DRY RUN" in out


def test_ensure_checkout_materializes_sparse_tree_and_is_idempotent(tmp_path, monkeypatch):
    import subprocess
    from oemseg import upstreams

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "tools").mkdir()
    (source / "config").mkdir()
    (source / "tools" / "train.py").write_text("print('ok')\n")
    (source / "config" / "model.py").write_text("model = {}\n")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()

    spec = upstreams.UpstreamSpec(
        name="tiny",
        repo=str(source),
        revision=revision,
        dest=tmp_path / "dest",
        sparse_paths=("tools", "config"),
        train_args=("tools/train.py", "config/model.py"),
        eval_args=("tools/train.py", "config/model.py"),
    )
    monkeypatch.setitem(upstreams.UPSTREAMS, "tiny", spec)
    monkeypatch.setitem(upstreams._ALIASES, "tiny", "tiny")

    dest = upstreams.ensure_checkout("tiny")
    assert (dest / "tools" / "train.py").exists()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip() == revision
    assert upstreams.ensure_checkout("tiny") == dest
