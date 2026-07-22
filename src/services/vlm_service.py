import asyncio
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
from io import BytesIO

from PIL import Image

from src.domain.models import VLMConfig
from src.utils.exceptions import AppError
from src.utils.logging_utils import log_vlm_error, log_vlm_success


DEFAULT_PROMPT_TEMPLATE = """\
役割: あなたは、提供されたメンバー名リストという知識ベースと、画像データという視覚情報を照合し、データ統合と正規化を行う専門のデータアナリストです。

【入力情報】

画像データ: 画像は別途画像ファイルとして提供されています。
メンバー名リスト (知識ベース): {member_list}
【処理指示】

データ抽出と正規化: 提供された画像から、最大で3名のユーザーに関連する情報（名前とファン数）を特定してください。
名前の正規化: 抽出した名前を、提供された「メンバー名リスト」と照合し、**表記揺れを完全に解消した「正規化された名前」**を生成してください。
最終データ抽出と出力: 正規化された名前をキーとし、対応するファン数を値として、以下のJSON形式で出力してください。
【出力形式】

```json

{{
  "正規化された名前A": [総獲得ファン数],
  "正規化された名前B": [総獲得ファン数],
  "正規化された名前C": [総獲得ファン数]
}}
```
【補足】
画像に含まれるユーザー（例：うん→ろん、しゃギア→シャギア・フロストなど）のデータについて、提供されたリストに基づき、重複や表記揺れを解消し、最大3名の関連データのみを抽出してください。
"""


class VLMService:
    """VLM (Vision-Language Model) による画像認識サービス

    llama-cli (subprocess) を用いて GGUF モデルにアクセスし、
    画像 + プロンプトからファン数を抽出する。
    """

    def __init__(self, config: VLMConfig, executor=None):
        self._executor = executor
        self.config = config
        if config.prompt_template:
            self._prompt_template = config.prompt_template
        elif Path("prompt_template/vlm_fan_count.txt").exists():
            self._prompt_template = Path("prompt_template/vlm_fan_count.txt").read_text(encoding="utf-8")
        else:
            self._prompt_template = DEFAULT_PROMPT_TEMPLATE
        self._member_list_text = self._load_member_list()

    def _load_member_list(self) -> str:
        """memberList.txt からメンバー名を読み込む"""
        try:
            member_file = Path("input/memberList.txt")
            if member_file.exists():
                names = member_file.read_text(encoding="utf-8").strip().split("\n")
                return "\n".join(f"  - {name.strip()}" for name in names if name.strip())
        except Exception:
            pass
        return "  （ファイルなし）"

    def _check_model_files(self):
        """モデルファイルの存在をチェック"""
        if not self.config.enabled:
            raise AppError(
                "VLM が無効です",
                "VLMConfig.enabled = True を設定してください"
            )

        model_path = Path(self.config.model_path)
        mmproj_path = Path(self.config.mmproj_path)

        print(f"[VLM] Checking model files...")
        print(f"[VLM] model_path: {model_path}")
        print(f"[VLM] mmproj_path: {mmproj_path}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_path}"
            )
        if not mmproj_path.exists():
            raise FileNotFoundError(
                f"マルチモーダルプロジェクトファイルが見つかりません: {mmproj_path}"
            )

        print(f"[VLM] Model files found.")

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """VLM 用の画像を前処理（コントラスト強調・解像度向上）"""
        from PIL import ImageEnhance
        
        # 解像度を 2 倍に拡大（テキストの読み取りを容易に）
        width, height = image.size
        new_size = (width * 2, height * 2)
        enhanced = image.resize(new_size, Image.LANCZOS)
        
        # コントラストを強化
        enhanced = ImageEnhance.Contrast(enhanced)
        enhanced = enhanced.enhance(2.0)
        
        # 明るさを調整
        enhanced = ImageEnhance.Brightness(enhanced)
        enhanced = enhanced.enhance(1.2)
        
        return enhanced

    def _run_llama_cli(self, image_path: Path, prompt: str, frame_index: int = None) -> str:
        """llama-cli を subprocess で実行し、画像認識結果を取得"""
        import time
        model_path = Path(self.config.model_path)
        mmproj_path = Path(self.config.mmproj_path)

        # プロンプトを一時ファイルに書き出し
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            # llama-cli のコマンドライン引数を構築
            cmd = [
                "llama\\llama-cli.exe",
                "-m", str(model_path),
                "--mmproj", str(mmproj_path),
                "--image", str(image_path),
                "-f", prompt_file,
                "--no-mmap",
                "--ctx-size", "32768",
                "--gpu-layers", "99",
                "--temp", "0.0",
                "--n-predict", "512",
                "--repeat-penalty", "1.1",
                "--top-k", "20",
            ]

            print(f"[VLM] Running llama-cli...")
            print(f"[VLM] Command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )

            try:
                stdout_data, _ = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout_data, _ = process.communicate(timeout=10)

            combined = stdout_data.decode('utf-8', errors='replace')

            # 生レスポンスをファイルに出力
            try:
                raw_dir = Path("output/debug/vlm_raw")
                raw_dir.mkdir(parents=True, exist_ok=True)
                frame_name = f"frame_{frame_index:04d}" if frame_index is not None else "unknown"
                raw_file = raw_dir / f"{frame_name}_raw.txt"
                raw_file.write_bytes(combined.encode('utf-8', errors='surrogateescape'))
                print(f"[VLM] Raw response saved to: {raw_file}")
            except Exception:
                pass

            safe = combined.encode('cp932', errors='replace').decode('cp932')
            print(f"[VLM] Output length: {len(combined)} chars", flush=True)
            print(f"[VLM] First 500 chars: {safe[:500]}", flush=True)

            return combined

        except Exception as e:
            print(f"[VLM] Error running llama-cli: {type(e).__name__}: {e}")
            raise
        finally:
            # 一時ファイルを削除
            try:
                Path(prompt_file).unlink(missing_ok=True)
            except Exception:
                pass

    async def analyze_image(self, image: Image.Image, frame_index: int = None) -> Dict[str, int]:
        """画像を解析し、ファン数の辞書を返す"""
        self._check_model_files()

        # 画像の前処理：コントラスト強調と解像度向上
        enhanced_image = self._enhance_image(image)
        original_size = image.size

        print(f"[VLM] Image size: {original_size}, Enhanced size: {enhanced_image.size}")

        try:
            enhanced_dir = Path("output/debug/vlm_enhanced")
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            frame_name = f"frame_{frame_index:04d}" if frame_index is not None else "unknown"
            enhanced_image.save(enhanced_dir / f"{frame_name}_enhanced.jpg", format="JPEG", quality=95)
            image.save(enhanced_dir / f"{frame_name}_original.jpg", format="JPEG", quality=95)
        except Exception:
            pass

        # 一時的に画像を保存
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir='output/debug/crop') as f:
            temp_image_path = f.name
        try:
            enhanced_image.save(temp_image_path, format="PNG")

            prompt = self._prompt_template.format(
                member_list=self._member_list_text,
            )
            print(f"[VLM] Prompt length: {len(prompt)} chars")

            # llama-cli で推論実行（同期処理を別スレッドで実行）
            loop = asyncio.get_running_loop()
            raw_text = await loop.run_in_executor(
                self._executor,
                lambda: self._run_llama_cli(Path(temp_image_path), prompt, frame_index)
            )

            print(f"[VLM] Raw response length: {len(raw_text) if raw_text else 0}")
            if not raw_text:
                print(f"[VLM] WARNING: VLM returned empty response for image size: {original_size}")

            result = self._parse_json_response(raw_text, frame_index=frame_index, image_size=original_size)

            # 成功時ログ
            log_vlm_success(
                frame_index=frame_index,
                raw_response=raw_text,
                prompt=prompt,
                parsed_result=result,
                image_size=original_size,
            )

            return result

        finally:
            # 一時画像ファイルを削除
            try:
                Path(temp_image_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _parse_json_response(self, text: str, frame_index: int = None, image_size: tuple = None) -> Dict[str, int]:
        """VLM の応答から JSON を抽出してパースする"""
        import re
        text = text.strip()

        # [End thinking] より後ろを切り取る（思考プロセスを除外）
        end_thinking_idx = text.find('[End thinking]')
        if end_thinking_idx != -1:
            text = text[end_thinking_idx + len('[End thinking]'):]

        # JSON コードブロックから抽出
        json_match = re.search(r'\`\`\`json\s*([\s\S]*?)\`\`\`', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 最初の { から最後の } までを抽出
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                raise AppError(
                    message=f"VLM の応答から JSON を抽出できませんでした。応答内容: {repr(text[:500])}",
                    hint="モデルの出力形式が期待と異なります。WinRT OCR モードをお試しください。"
                )

        safe = json_str[:300].encode('cp932', errors='replace').decode('cp932')
        print(f"[VLM] Extracted JSON: {safe}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[VLM] JSON parse error: {e}")
            safe = repr(text[:500]).encode('cp932', errors='replace').decode('cp932')
            print(f"[VLM] Raw text: {safe}")
            log_vlm_error(
                error=e,
                raw_response=text,
                extracted_json=json_str,
                frame_index=frame_index,
                image_size=image_size,
            )
            raise AppError(
                message=f"VLM の応答を JSON としてパースできませんでした。応答内容: {repr(text[:500])}",
                hint="モデルの精度が低い、または画像が読み取りにくい場合があります。WinRT OCR モードをお試しください。"
            )

        if not isinstance(data, dict):
            print(f"[VLM] WARNING: Expected dict, got {type(data).__name__}")
            log_vlm_error(
                error=TypeError(f"Expected dict, got {type(data).__name__}"),
                raw_response=text,
                extracted_json=json_str,
                frame_index=frame_index,
                image_size=image_size,
            )
            raise AppError(
                message=f"VLM の応答が辞書形式ではありません。型: {type(data).__name__}",
                hint="モデルの出力形式が期待と異なります。WinRT OCR モードをお試しください。"
            )

        result = {}
        for key, value in data.items():
            # キーの正規化：空白・改行を除去
            clean_key = key.strip() if isinstance(key, str) else str(key).strip()
            if not clean_key:
                print(f"[VLM] Skipping empty key")
                continue

                safe = f"Processing key={repr(clean_key)}, value={value!r}, type={type(value).__name__}".encode('cp932', errors='replace').decode('cp932')
                print(f"[VLM] {safe}")

            try:
                if isinstance(value, (int, float)):
                    result[clean_key] = int(value)
                elif isinstance(value, str):
                    cleaned = value.replace(',', '').replace(' ', '')
                    try:
                        result[clean_key] = int(cleaned)
                    except ValueError:
                        print(f"[VLM] Cannot convert string value to int: {repr(value)}")
                elif isinstance(value, list):
                    # リスト形式 [数値] の対応
                    if len(value) == 1:
                        item = value[0]
                        if isinstance(item, (int, float)):
                            result[clean_key] = int(item)
                        elif isinstance(item, str):
                            cleaned = item.replace(',', '').replace(' ', '')
                            try:
                                result[clean_key] = int(cleaned)
                            except ValueError:
                                print(f"[VLM] Cannot convert list item to int: {repr(item)}")
                    elif len(value) > 1:
                        # 複数の値がある場合は合計
                        total = 0
                        for item in value:
                            if isinstance(item, (int, float)):
                                total += item
                            elif isinstance(item, str):
                                cleaned = item.replace(',', '').replace(' ', '')
                                try:
                                    total += int(cleaned)
                                except ValueError:
                                    pass
                        result[clean_key] = int(total)
                    else:
                        print(f"[VLM] Empty list value for key: {clean_key}")
                elif isinstance(value, dict):
                    # 入れ子辞書の対応（例: {"count": 123}）
                    if 'count' in value:
                        result[clean_key] = int(value['count'])
                    elif 'value' in value:
                        result[clean_key] = int(value['value'])
                    else:
                        print(f"[VLM] Skipping nested dict without count/value: {value}")
                else:
                    print(f"[VLM] Skipping unsupported value type: {type(value).__name__}")
            except Exception as e:
                print(f"[VLM] Error processing key {clean_key}: {e}")
                log_vlm_error(
                    error=e,
                    extracted_json=json.dumps(result, ensure_ascii=False),
                    frame_index=frame_index,
                    image_size=image_size,
                )
                continue

        print(f"[VLM] Parsed result: {result}")
        return result

    def _safe_print(self, msg: str) -> None:
        """Unicode safe print (cp932 fallback)"""
        safe = msg.encode('cp932', errors='replace').decode('cp932')
        print(f"[VLM] {safe}")

    async def stop(self):
        """VLM リソースを解放する"""
        pass
