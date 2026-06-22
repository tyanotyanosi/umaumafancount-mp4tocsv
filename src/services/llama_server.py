import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import aiohttp


class LlamaServer:
    """llama-server をサブプロセスとして起動・管理する"""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        port: int = 8080,
        ngl: int = 99,
        debug: bool = False,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.port = port
        self.ngl = ngl
        self.debug = debug
        self.base_url = f"http://127.0.0.1:{port}"
        self._process: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = None

    async def start(self):
        """llama-server を起動"""
        cmd = [
            str(Path("llama", "llama-server.exe")),
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--ctx-size", "32768",
            "-ngl", str(self.ngl),
            "--reasoning", "off",
            "--batch-size", "1024",
        ]
        print(f"Starting llama-server: {' '.join(cmd)}")

        log_dir = Path("output/debug/llama_server")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "server.log"

        if self.debug:
            self._process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True,
                bufsize=1,
            )
        else:
            # llama-server のログをファイルに保存
            self._log_file = log_file
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            threading.Thread(target=self._feed_stdout, daemon=True).start()
            threading.Thread(target=self._feed_stderr, daemon=True).start()

        # サーバーが起動するまで待機
        await self._wait_for_server()
        
        # HTTP セッションを初期化
        self._session_lock = asyncio.Lock()
        connector = aiohttp.TCPConnector(limit=10, force_close=False)
        self._session = aiohttp.ClientSession(connector=connector)

    def _feed_stdout(self):
        """llama-server の stdout をリアルタイムで出力（デバッグモード外）"""
        try:
            with open(self._log_file, "a", encoding="utf-8") as log_f:
                for line in iter(self._process.stdout.readline, ""):
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_f.write(line)
                        log_f.flush()
        except Exception:
            pass

    def _feed_stderr(self):
        """llama-server の stderr をリアルタイムで出力（デバッグモード外）"""
        try:
            with open(self._log_file, "a", encoding="utf-8") as log_f:
                for line in iter(self._process.stderr.readline, ""):
                    if line:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                        log_f.write(line)
                        log_f.flush()
        except Exception:
            pass

    async def _wait_for_server(self, timeout: int = 60):
        """サーバーが応答するまで待機"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as resp:
                        if resp.status == 200:
                            print("llama-server started successfully")
                            return
            except Exception as e:
                elapsed = int(time.time() - start_time)
                print(f"[llama-server] waiting for server... ({elapsed}s) - {type(e).__name__}")
            await asyncio.sleep(1)
        raise RuntimeError("llama-server failed to start within timeout")

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP セッションを取得（なければ作成）"""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(limit=10, force_close=False)
                    self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def chat_completion(
        self,
        prompt: str,
        image_b64: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """チャット補完を HTTP で実行（llama-server mtmd 形式）"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        print(f"[llama-server] Sending request to {self.base_url}/v1/chat/completions")
        print(f"[llama-server] Prompt length: {len(prompt)} chars")
        print(f"[llama-server] Image base64 length: {len(image_b64)} chars")
        print(f"[llama-server] Total payload size (est): {len(json.dumps(payload))} bytes")
        
        import base64 as b64mod
        try:
            img_data = b64mod.b64decode(image_b64)
            print(f"[llama-server] Decoded image size: {len(img_data)} bytes")
            if len(img_data) > 0:
                print(f"[llama-server] JPEG header bytes: {img_data[:4].hex()}")
                if img_data[:2] == b'\xff\xd8':
                    print(f"[llama-server] JPEG signature: valid")
                else:
                    print(f"[llama-server] WARNING: JPEG signature invalid, first bytes: {img_data[:10].hex()}")
        except Exception as e:
            print(f"[llama-server] WARNING: Failed to decode base64: {e}")

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[llama-server] Error response (status={resp.status}): {error_text[:500]}")
                    raise RuntimeError(f"llama-server error: {error_text}")
                result = await resp.json()
                print(f"[llama-server] Response keys: {list(result.keys())}")
                choices = result.get("choices", [])
                print(f"[llama-server] Number of choices: {len(choices)}")
                if choices:
                    message = choices[0].get("message", {})
                    print(f"[llama-server] Message keys: {list(message.keys())}")
                    content = message.get("content", "")
                    print(f"[llama-server] Content length: {len(content) if content else 0}")
                    if not content:
                        reasoning = message.get("reasoning_content", "")
                        if reasoning:
                            print(f"[llama-server] VLM used reasoning_content. Length: {len(reasoning)}")
                            print(f"[llama-server] Reasoning content preview: {reasoning[:200]}...")
                            content = reasoning
                    return content or ""
                return ""
        except aiohttp.TimeoutError:
            print(f"[llama-server] Request timed out after 120s")
            raise
        except Exception as e:
            print(f"[llama-server] Request failed: {type(e).__name__}: {e}")
            raise

    async def stop(self):
        """llama-server を停止"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._process and self._process.poll() is None:
            print("Stopping llama-server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
