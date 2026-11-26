"""Generate a small, clearly labeled synthetic dataset matching CERT r4.2's schema.

This synthetic sample is ONLY for testing pipeline functionality (python main.py).
It is NOT real CERT r4.2 data and must never be reported as real experimental evidence.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import config


def generate_synthetic_sample():
    rng = np.random.default_rng(config.RANDOM_SEED)
    out_dir = config.DATA_SAMPLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_users = 40
    n_days = 100
    start_date = datetime(2010, 1, 4)  # Monday
    users = [f"SYN{i:04d}" for i in range(1, n_users + 1)]

    # Pick 9 synthetic insiders spread across train, validation, and test date ranges
    insider_attack_days = {
        "SYN0003": 22,
        "SYN0007": 35,
        "SYN0011": 48,
        "SYN0015": 55,
        "SYN0019": 67,
        "SYN0023": 74,
        "SYN0027": 83,
        "SYN0031": 90,
        "SYN0035": 95,
    }

    logon_rows = []
    device_rows = []
    file_rows = []
    email_rows = []
    http_rows = []
    psych_rows = []
    insider_rows = []

    row_id = 1
    for user in users:
        psych_rows.append({
            "employee_name": f"User {user}",
            "user_id": user,
            "O": int(rng.integers(15, 50)),
            "C": int(rng.integers(15, 50)),
            "E": int(rng.integers(15, 50)),
            "A": int(rng.integers(15, 50)),
            "N": int(rng.integers(15, 50)),
        })

    for user, attack_day in insider_attack_days.items():
        attack_start = start_date + timedelta(days=attack_day, hours=21)
        attack_end = attack_start + timedelta(days=2)
        insider_rows.append({
            "dataset": "4.2",
            "scenario": int(rng.choice([1, 2, 3])),
            "details": f"synthetic_{user}.csv",
            "user": user,
            "start": attack_start.strftime("%m/%d/%Y %H:%M:%S"),
            "end": attack_end.strftime("%m/%d/%Y %H:%M:%S"),
        })

    normal_urls = ["http://intranet.dtaa.com/home", "http://news.example.com/tech", "http://docs.example.com/ref"]
    job_urls = ["http://www.indeed.com/jobs", "http://www.linkedin.com/jobs", "http://www.monster.com/careers"]
    upload_urls = ["http://wikileaks.org/upload", "http://www.dropbox.com/upload", "http://pastebin.com/raw"]

    for day_idx in range(n_days):
        cur_date = start_date + timedelta(days=day_idx)
        is_weekend = cur_date.weekday() >= 5

        for user in users:
            pc = f"PC-{user[-4:]}"
            attack_day = insider_attack_days.get(user, None)
            in_pre_window = (
                attack_day is not None
                and (attack_day - config.PREDICTION_HORIZON_DAYS) <= day_idx < attack_day
            )

            if is_weekend and not in_pre_window and rng.random() < 0.85:
                continue

            # Normal daily logons
            n_logons = 1 if rng.random() < 0.85 else 2
            for _ in range(n_logons):
                hour = int(rng.integers(8, 17))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                logon_rows.append({
                    "id": f"L{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "activity": "Logon",
                })
                row_id += 1

            # After-hours logon
            if (in_pre_window and rng.random() < 0.65) or (not in_pre_window and rng.random() < 0.04):
                hour = int(rng.choice([20, 21, 22, 23, 5, 6]))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                logon_rows.append({
                    "id": f"L{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "activity": "Logon",
                })
                row_id += 1

            # USB device connects
            usb_count = int(rng.poisson(1.8)) if in_pre_window else int(rng.poisson(0.08))
            for _ in range(usb_count):
                if in_pre_window and rng.random() < 0.6:
                    hour = int(rng.choice([20, 21, 22, 23]))
                elif rng.random() < 0.15:
                    hour = int(rng.choice([20, 21]))
                else:
                    hour = int(rng.integers(9, 17))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                device_rows.append({
                    "id": f"D{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "activity": "Connect",
                })
                row_id += 1

            # File accesses
            file_count = int(rng.poisson(6.0)) if in_pre_window else int(rng.poisson(1.2))
            for _ in range(file_count):
                hour = int(rng.choice([20, 21, 22])) if (in_pre_window and rng.random() < 0.5) else int(rng.integers(9, 17))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                file_rows.append({
                    "id": f"F{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "filename": "doc.pdf", "content": "",
                })
                row_id += 1

            # Emails
            email_count = int(rng.poisson(4.0)) if in_pre_window else int(rng.poisson(2.0))
            for _ in range(email_count):
                hour = int(rng.integers(8, 19))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                is_external = (in_pre_window and rng.random() < 0.45) or (not in_pre_window and rng.random() < 0.08)
                recipient = "external@gmail.com" if is_external else f"colleague{config.COMPANY_DOMAIN}"
                attachments = int(rng.poisson(1.2)) if is_external and in_pre_window else int(rng.random() < 0.15)
                email_rows.append({
                    "id": f"E{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "to": recipient, "cc": "", "bcc": "",
                    "from": f"{user.lower()}{config.COMPANY_DOMAIN}",
                    "size": int(rng.integers(5000, 500000)), "attachments": attachments, "content": "",
                })
                row_id += 1

            # HTTP web visits
            http_count = int(rng.poisson(6.0)) if in_pre_window else int(rng.poisson(3.0))
            for _ in range(http_count):
                if in_pre_window and rng.random() < 0.35:
                    url = str(rng.choice(job_urls))
                    hour = int(rng.choice([12, 17, 20, 21]))
                elif in_pre_window and rng.random() < 0.25:
                    url = str(rng.choice(upload_urls))
                    hour = int(rng.choice([20, 21, 22]))
                else:
                    url = str(rng.choice(normal_urls))
                    hour = int(rng.integers(8, 18))
                ts = cur_date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                http_rows.append({
                    "id": f"H{row_id}", "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                    "user": user, "pc": pc, "url": url, "content": "",
                })
                row_id += 1

    pd.DataFrame(logon_rows).to_csv(out_dir / "logon.csv", index=False)
    pd.DataFrame(device_rows).to_csv(out_dir / "device.csv", index=False)
    pd.DataFrame(file_rows).to_csv(out_dir / "file.csv", index=False)
    pd.DataFrame(email_rows).to_csv(out_dir / "email.csv", index=False)
    pd.DataFrame(http_rows).to_csv(out_dir / "http.csv", index=False)
    pd.DataFrame(psych_rows).to_csv(out_dir / "psychometric.csv", index=False)
    pd.DataFrame(insider_rows).to_csv(out_dir / "insiders.csv", index=False)

    (out_dir / "README.md").write_text(
        "# Synthetic Sample Data\n\n"
        "All CSV files in this folder are **100% synthetic** and generated by "
        "`src/generate_sample.py` using `RANDOM_SEED = 42`.\n\n"
        "They match the column schema of CMU CERT r4.2 solely to allow "
        "`python main.py` to run end-to-end without redistributing licensed CERT data. "
        "Metrics produced on this folder verify code execution only and do not represent "
        "performance on the real CERT r4.2 dataset.\n",
        encoding="utf-8",
    )
    print(f"Synthetic sample generated in {out_dir}")


if __name__ == "__main__":
    generate_synthetic_sample()