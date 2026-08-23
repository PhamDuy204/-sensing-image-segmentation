"""Optional end-of-run email notification."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def validate_email_settings(args) -> None:
    if not args.notify_email:
        return
    if not args.smtp_user:
        raise ValueError("--smtp-user is required with --notify-email")
    if not os.getenv(args.smtp_password_env):
        raise ValueError(
            f"environment variable {args.smtp_password_env!r} is required with --notify-email"
        )


def send_training_email(args, stats: dict[str, object]) -> None:
    if not args.notify_email:
        return
    password = os.environ[args.smtp_password_env]
    message = EmailMessage()
    message["Subject"] = f"OEM training complete: {stats['run_name']}"
    message["From"] = args.smtp_from or args.smtp_user
    message["To"] = args.notify_email
    message.set_content(
        "\n".join(
            [
                f"run: {stats['run_name']}",
                f"best val mIoU: {stats['best_val_miou']:.6f} (epoch {stats['best_val_epoch']})",
                f"best observed test mIoU: {stats['best_test_miou']:.6f} (epoch {stats['best_test_epoch']})",
                f"final test mIoU: {stats['final_test_miou']:.6f}",
                f"output: {stats['output']}",
            ]
        )
    )
    with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=30) as client:
        if not args.smtp_no_starttls:
            client.starttls()
        client.login(args.smtp_user, password)
        client.send_message(message)
