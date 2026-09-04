from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_default_evaluation_schedule():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    factory = load(ROOT / "oemseg/schedulers/factory.py", "schedule_factory")
    plan = [factory.evaluation_schedule(e, 45, 30, 1, 3) for e in range(1, 46)]
    assert all(x == (False, False) for x in plan[:30])
    assert plan[30] == (True, False)
    assert plan[31] == (True, False)
    assert plan[32] == (True, True)
    assert plan[43] == (True, False)
    assert plan[44] == (True, True)


def test_final_epoch_forces_val_and_test():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    factory = load(ROOT / "oemseg/schedulers/factory.py", "schedule_factory_final")
    assert factory.evaluation_schedule(32, 32, 30, 1, 3) == (True, True)


def test_launcher_validates_gpu_model_counts_and_builds_commands():
    launch = load(ROOT / "scripts/launch.py", "oem_launch")
    assert launch.parse_gpu_ids("0,2", [0, 1, 2]) == [0, 2]
    try:
        launch.parse_gpu_ids("0,3", [0, 1, 2])
    except ValueError as exc:
        assert "available" in str(exc).lower()
    else:
        raise AssertionError("missing GPU must fail")
    try:
        launch.validate_parallel([0, 1], ["unet"])
    except ValueError as exc:
        assert "one model per gpu" in str(exc).lower()
    else:
        raise AssertionError("GPU/model mismatch must fail")
    command = launch.build_distributed_command([0, 1], "mambavision", ["--epochs", "1"])
    assert "accelerate.commands.launch" in command
    assert command[command.index("--num_processes") + 1] == "2"
    assert command[-4:] == ["--model", "mambavision", "--epochs", "1"]


def test_setup_env_uses_prebuilt_mamba_wheel_without_source_fallback():
    setup = (ROOT / "scripts/setup_env.sh").read_text()
    assert "releases/download/v${MAMBA_VERSION}" in setup
    assert "mamba_ssm-${MAMBA_VERSION}+${MAMBA_TAG}" in setup
    assert 'pip install --no-deps "$MAMBA_WHEEL_URL"' in setup
    assert "MAMBA_FORCE_BUILD" not in setup


def test_email_uses_password_env_and_smtp():
    notifications = load(ROOT / "oemseg/utils/notifications.py", "oem_notifications")
    args = SimpleNamespace(
        notify_email="dest@example.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="runner@example.com",
        smtp_from=None,
        smtp_password_env="SMTP_PASSWORD",
        smtp_no_starttls=False,
    )
    stats = {
        "run_name": "unet-1",
        "best_val_epoch": 40,
        "best_val_miou": 0.72,
        "best_test_epoch": 45,
        "best_test_miou": 0.69,
        "final_test_miou": 0.68,
        "output": "/tmp/out",
    }
    with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False), patch(
        "smtplib.SMTP"
    ) as smtp_cls:
        notifications.send_training_email(args, stats)
    smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=30)
    client = smtp_cls.return_value.__enter__.return_value
    client.starttls.assert_called_once()
    client.login.assert_called_once_with("runner@example.com", "secret")
    message = client.send_message.call_args.args[0]
    assert message["To"] == "dest@example.com"
    assert "best val mIoU: 0.720000 (epoch 40)" in message.get_content()


def test_readme_documents_q1_paper_models_and_protocol_boundaries():
    readme = (ROOT / "README.md").read_text()
    for name in ("PyramidMamba", "GeoSA-BaSA", "HG-RSOVSSeg", "RepSTDC"):
        assert name in readme
    assert "python scripts/paper_models.py setup all" in readme
    assert "python train.py --model pyramidmamba" in readme
    assert "repstdc-ca_512x512_80k_oem.py" in readme
    assert "official upstream protocol" in readme.lower()
    assert "not directly comparable" in readme.lower()
