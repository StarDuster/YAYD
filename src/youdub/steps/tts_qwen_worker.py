from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from loguru import logger

from ..config import Settings
from ..interrupts import CancelledByUser, check_cancelled
from ..models import ModelCheckError


_QWEN_WORKER_READY = "__READY__"
_QWEN_WORKER_STUB_ENV = "YOUDUB_QWEN_WORKER_STUB"


def _env_flag(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_qwen_worker_script_path() -> Path:
    # Project layout (repo checkout):
    # - <repo>/scripts/qwen_tts_worker.py
    # - <repo>/src/youdub/steps/tts_qwen_worker.py
    #
    # Resolve robustly by walking upwards so refactors won't break the path.
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "scripts" / "qwen_tts_worker.py"
        if cand.exists():
            return cand
    # Best-effort fallback: assume current working directory is repo root.
    return Path.cwd() / "scripts" / "qwen_tts_worker.py"


class _QwenTtsWorker:
    def __init__(self, python_exe: str, model_path: str, stub: bool = False):
        script = _get_qwen_worker_script_path()
        if not script.exists():
            raise ModelCheckError(f"未找到 Qwen3-TTS worker 脚本: {script}")

        cmd = [python_exe, "-u", str(script), "--model-path", model_path]
        if stub:
            cmd.append("--stub")

        logger.info(f"启动Qwen3-TTS工作进程 (stub={stub})")
        logger.debug(f"Qwen3-TTS工作进程命令: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise ModelCheckError(f"启动 Qwen3-TTS worker 失败: {exc}") from exc

        logger.info(f"Qwen3-TTS工作进程 pid={self._proc.pid}，等待就绪...")

        self._stderr_tail: deque[str] = deque(maxlen=200)

        def _drain_stderr() -> None:
            stream = self._proc.stderr
            if stream is None:
                return
            try:
                for line in stream:
                    s = line.rstrip("\n")
                    if s:
                        self._stderr_tail.append(s)
            except Exception:
                # Best-effort: avoid crashing main flow due to stderr reader issues.
                return

        threading.Thread(target=_drain_stderr, daemon=True).start()

        startup_begin = time.monotonic()
        startup_done = threading.Event()
        startup_timed_out = threading.Event()
        try:
            startup_timeout_sec = float(os.getenv("YOUDUB_QWEN_WORKER_STARTUP_TIMEOUT_SEC", "1800") or "1800")
        except Exception:
            startup_timeout_sec = 1800.0

        def _startup_heartbeat() -> None:
            # 避免“正在加载但完全没日志输出”的错觉
            while not startup_done.wait(10.0):
                elapsed = time.monotonic() - startup_begin
                logger.info(f"Qwen3-TTS工作进程加载中... ({elapsed:.1f}秒)")

        def _startup_watchdog() -> None:
            if startup_timeout_sec <= 0:
                return
            if startup_done.wait(startup_timeout_sec):
                return
            startup_timed_out.set()
            logger.error(f"Qwen3-TTS工作进程启动超时 {startup_timeout_sec:.0f}秒，正在终止。")
            try:
                self._proc.terminate()
            except Exception:
                return

        threading.Thread(target=_startup_heartbeat, daemon=True).start()
        threading.Thread(target=_startup_watchdog, daemon=True).start()

        assert self._proc.stdout is not None
        startup_stdout_tail: list[str] = []
        ready_ok = False
        try:
            while True:
                check_cancelled()
                line = self._read_stdout_line(timeout_sec=0.2)
                if line is None:
                    if self._proc.poll() is not None:
                        break
                    continue
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if s == _QWEN_WORKER_READY:
                    ready_ok = True
                    break
                startup_stdout_tail.append(s)
                if len(startup_stdout_tail) > 50:
                    startup_stdout_tail = startup_stdout_tail[-50:]
        except CancelledByUser:
            self.close()
            raise
        finally:
            startup_done.set()

        if ready_ok:
            elapsed = time.monotonic() - startup_begin
            logger.info(f"Qwen3-TTS工作进程就绪，耗时 {elapsed:.1f}秒")

        if not ready_ok:
            rc = self._proc.poll()
            stdout_tail = "\n".join(startup_stdout_tail).strip()
            stderr_tail = "\n".join(list(self._stderr_tail)).strip()
            self.close()

            details: list[str] = []
            if startup_timed_out.is_set():
                details.append(f"startup_timeout_sec={startup_timeout_sec:.0f}")
            if rc is not None:
                details.append(f"exit_code={rc}")
                if rc == -9:
                    details.append("看起来像是被 SIGKILL 杀掉（常见原因：内存不足/OOM 或 WSL2 内存限制）")
            if stdout_tail:
                details.append("stdout:\n" + stdout_tail)
            if stderr_tail:
                details.append("stderr:\n" + stderr_tail)

            extra = ("\n" + "\n".join(details)) if details else ""
            raise ModelCheckError(f"Qwen3-TTS worker 启动失败。{extra}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "_QwenTtsWorker":
        py = settings.resolve_path(settings.qwen_tts_python_path)
        if not py or not py.exists():
            raise ModelCheckError(
                f"未找到 Qwen3-TTS Python: {py}。请设置 QWEN_TTS_PYTHON 为有效的 python 可执行文件（该环境需安装 qwen-tts）。"
            )

        model_dir = settings.resolve_path(settings.qwen_tts_model_path)
        if not model_dir or not model_dir.exists():
            raise ModelCheckError(f"未找到 Qwen3-TTS 模型目录: {model_dir}。请先下载模型权重并设置 QWEN_TTS_MODEL_PATH。")

        stub = os.getenv(_QWEN_WORKER_STUB_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        return cls(python_exe=str(py), model_path=str(model_dir), stub=stub)

    def _read_stdout_line(self, timeout_sec: float = 0.2) -> str | None:
        """Read one line from worker stdout with a small timeout.

        On Linux, use select() so Ctrl+C cancellation can be observed promptly.
        On Windows, fall back to blocking readline(); Ctrl+C will typically also stop the child.
        """
        assert self._proc.stdout is not None
        if os.name == "nt":
            return self._proc.stdout.readline()

        # IMPORTANT:
        # subprocess.Popen(..., text=True) wraps stdout in TextIOWrapper with its own buffer.
        # If a previous readline() pulled multiple lines into the internal buffer, the OS-level fd
        # may have no pending bytes, causing select(fd) to return empty while data is still available
        # in the Python buffer. This can deadlock handshake (e.g. "__READY__" already buffered).
        try:
            buf = getattr(self._proc.stdout, "buffer", None)
            if buf is not None and hasattr(buf, "peek"):
                # NOTE:
                # BufferedReader.peek(n>0) may block while trying to fill from the raw pipe.
                # We only want to check *already buffered* bytes, so use peek(0) here.
                if buf.peek(0):
                    return self._proc.stdout.readline()
        except Exception:
            pass

        try:
            import select

            fd = self._proc.stdout.fileno()
            r, _w, _x = select.select([fd], [], [], max(0.0, float(timeout_sec)))
            if not r:
                return None
        except Exception:
            # Fallback: best-effort blocking read.
            return self._proc.stdout.readline()
        return self._proc.stdout.readline()

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
        language: str = "Auto",
        *,
        timeout_sec: float = 180.0,
    ) -> dict:
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen3-TTS worker 已退出")

        req = {
            "cmd": "synthesize",
            "text": text,
            "language": language,
            "speaker_wav": speaker_wav,
            "output_path": output_path,
        }

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        skipped: list[str] = []
        max_skip = 50
        start_time = time.monotonic()
        last_heartbeat = start_time
        while True:
            check_cancelled()
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                # Worker may be stuck in GPU inference; try hard to terminate it.
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Qwen3-TTS worker 超时 ({timeout_sec:.0f}秒)，"
                    f"text={text[:30]}...，已等待 {elapsed:.1f}秒。{extra}"
                )
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                logger.info(f"Qwen3-TTS 等待中... ({elapsed:.1f}s / {timeout_sec:.0f}s)")
            line = self._read_stdout_line(timeout_sec=0.2)
            if line is None:
                if self._proc.poll() is not None:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    raise RuntimeError("Qwen3-TTS worker 已退出" + extra)
                continue
            if not line:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                raise RuntimeError("Qwen3-TTS worker 无输出" + extra)

            s = line.strip()
            if not s:
                continue

            try:
                resp = json.loads(s)
                break
            except json.JSONDecodeError:
                skipped.append(s)
                if len(skipped) >= max_skip:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    noise = "\n".join(skipped[-10:])
                    raise RuntimeError(
                        "Qwen3-TTS worker 输出无法解析为 JSON（协议被日志污染）。" f"\nstdout_tail:\n{noise}{extra}"
                    )
                continue

        if not resp.get("ok"):
            err = str(resp.get("error", "unknown error"))
            trace = resp.get("trace")
            if trace:
                raise RuntimeError(err + "\n" + str(trace))
            raise RuntimeError(err)
        return resp

    def synthesize_batch(self, items: list[dict], timeout_sec: float = 300.0) -> dict:
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen3-TTS worker 已退出")

        req = {
            "cmd": "synthesize_batch",
            "items": items,
        }

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        skipped: list[str] = []
        max_skip = 50
        start_time = time.monotonic()
        last_heartbeat = start_time
        while True:
            check_cancelled()
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                # Worker may be stuck in GPU inference; try hard to terminate it.
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Qwen3-TTS worker 超时 ({timeout_sec:.0f}秒)，"
                    f"batch_size={len(items)}，已等待 {elapsed:.1f}秒。"
                    f"可能是 GPU 推理卡住或显存不足。{extra}"
                )
            # Heartbeat log every 30s to show we're still waiting
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                logger.info(f"Qwen3-TTS 批处理等待中... ({elapsed:.1f}s / {timeout_sec:.0f}s) batch_size={len(items)}")
            line = self._read_stdout_line(timeout_sec=0.2)
            if line is None:
                if self._proc.poll() is not None:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    raise RuntimeError("Qwen3-TTS worker 已退出" + extra)
                continue
            if not line:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                raise RuntimeError("Qwen3-TTS worker 无输出" + extra)

            s = line.strip()
            if not s:
                continue

            try:
                resp = json.loads(s)
                break
            except json.JSONDecodeError:
                skipped.append(s)
                if len(skipped) >= max_skip:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    noise = "\n".join(skipped[-10:])
                    raise RuntimeError(
                        "Qwen3-TTS worker 输出无法解析为 JSON（协议被日志污染）。" f"\nstdout_tail:\n{noise}{extra}"
                    )
                continue

        if not resp.get("ok"):
            err = str(resp.get("error", "unknown error"))
            trace = resp.get("trace")
            if trace:
                raise RuntimeError(err + "\n" + str(trace))
            raise RuntimeError(err)
        return resp

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if not proc:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        # Close stdin early so the worker can exit its read loop.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

