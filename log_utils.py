"""
log_utils.py — 범용 학습 로거

백그라운드 실행 시 Python stdout 버퍼링 문제 해결:
  - print(flush=True) → stdout 즉시 플러시
  - open(buffering=1)  → 파일에 한 줄씩 즉시 기록

Usage:
    from log_utils import Logger

    with Logger("world_model/results/ssm_xxx") as logger:
        logger.log(f"Epoch {e}/60  tr={tr:.4f}  val={val:.4f}")

    # out_dir 없으면 CWD/logs/ 에 자동 생성
    logger = Logger()
    logger.log("hello")
    logger.close()
"""

import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """
    stdout + 파일 동시 로거.

    Args:
        out_dir: 로그 파일이 저장될 디렉토리. None이면 ./logs/.
        filename: 로그 파일명. None이면 train_{timestamp}.log.
        timestamps: 각 줄 앞에 [HH:MM:SS] 붙일지 여부.
    """

    def __init__(
        self,
        out_dir: "str | Path | None" = None,
        filename: "str | None" = None,
        timestamps: bool = False,
    ):
        self.timestamps = timestamps

        if out_dir is None:
            out_dir = Path.cwd() / "logs"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"train_{ts}.log"
        self.log_path = self.out_dir / filename

        # buffering=1 → line-buffered: 줄 끝마다 자동 flush
        self._fh = open(self.log_path, "w", buffering=1, encoding="utf-8")

    # ── 로깅 ──────────────────────────────────────────────────────────────────

    def log(self, msg: str = "") -> None:
        if self.timestamps:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
        else:
            line = msg
        print(line, flush=True)
        self._fh.write(line + "\n")

    def section(self, title: str, width: int = 60) -> None:
        sep = "─" * width
        self.log(sep)
        self.log(title)
        self.log(sep)

    # ── 리소스 관리 ───────────────────────────────────────────────────────────

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
