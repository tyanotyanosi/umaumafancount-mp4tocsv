# ocr_winrt_safe.py
import sys
import asyncio
from io import BytesIO
from PIL import Image
import cv2
from typing import Iterable, Tuple, Optional, List
import numpy as np
import pathlib
import os 
import glob
import shutil
from tqdm import tqdm
import re
import json
import argparse
from pathlib import Path
# --- Windows の一部環境で念のため ---
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# WinRT (PyWinRT)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Umamusume Pretty Derby の動画からファン数を抽出するツール(WinRt OCR使用)")
    p.add_argument("video", type=str, help="入力動画のファイル名(例：UmamusumePrettyDerby_Jpn*.mp4)")
    p.add_argument("--debug", action="store_true", help="中間ファイルを削除せずに残す")
    p.add_argument("--img-scale", type=str, default=None, help="グレースケールで文字認識する場合は'gray'を指定")
    return p.parse_args()


def cleanup(base_path, debug_path):
    shutil.rmtree(base_path, ignore_errors=True)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(debug_path / "crop", exist_ok=True)
    os.makedirs(debug_path / "gray", exist_ok=True)
    os.makedirs(debug_path / "frames", exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(debug_path/ "text", exist_ok=True)
def save_all_frames(video_path, dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            break
    cap.release()
def _crop_image(img, dir_path, basename,ext='png'):
    h, w = img.shape[:2]
    cropped_img = img[int(h*0.45):int(h*0.88), int(w*0.15):int(w*0.45)]
    #切り抜きたい部分の座標を入力

    base_path = os.path.join(dir_path, basename)
    cv2.imwrite(f'{base_path}.{ext}',cropped_img)
    cv2.waitKey(0)
    return f'{base_path}.{ext}'
async def _ocr_image_bytes(img_bytes: bytes) -> str:
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(img_bytes)
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)

    decoder = await BitmapDecoder.create_async(stream)
    software_bitmap = await decoder.get_software_bitmap_async()

    engine = OcrEngine.try_create_from_user_profile_languages()  # OSの優先言語
    result = await engine.recognize_async(software_bitmap)
    return result.text

def ocr_with_winrt(image_path: str) -> str:
    # PILで読み、PNGに正規化してから WinRT に渡す
    with Image.open(image_path) as im:
        buf = BytesIO()
        im.save(buf, format="PNG")
        return asyncio.run(_ocr_image_bytes(buf.getvalue()))

def to_gray(path, dir_path, basename,ext='png'):
    im=cv2.imread(path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    base_path = os.path.join(dir_path, basename)
    cv2.imwrite(f'{base_path}.{ext}', im_gray)
    return f'{base_path}.{ext}'

def get_fan_count(texts, menber, fans):
    


    token_boundary = r"[0-9A-Za-z\u3040-\u30FF\u3400-\u9FFF]"

    pattern = re.compile(
        rf"(?<!{token_boundary})(?<![\d,])"   # 直前がトークン文字 or [数字/カンマ] なら不一致
        r"\d{1,3}(?:,\d{3})+"                 # カンマ区切りの数値（カンマ必須）
        rf"(?!{token_boundary})(?![\d,])"     # 直後がトークン文字 or [数字/カンマ] なら不一致
    )


    fancounts = {}
    for text in texts:
        if menber in text:
            match = pattern.search(text)
            if match:
                value = match.group()              # '3,249,444,186'
                numeric_value = int(value.replace(',', ''))  # 3249444186
                if numeric_value in fans:
                    continue
                if numeric_value in fancounts.keys():
                    fancounts[numeric_value] = fancounts[numeric_value] + 1
                else:
                    fancounts[numeric_value] = 1
    if len(fancounts) != 0:
        sorted_dict  = sorted(fancounts.items(), key=lambda x: x[1],reverse = True)
        return sorted_dict[0][0]
    return None

if __name__ == "__main__":
    args = parse_args()
    base_path = Path("output/")
    debug_path = base_path / "debug"
    print("=== CleanUP ===")
    cleanup = cleanup(base_path, debug_path)
    video_path = Path(args.video)
    print(f"=== Input video: {video_path} ===")
    print("Parse video and save frames...")
    
    save_all_frames(video_path, debug_path, "frames/frame", "png")
    images = glob.glob(f"{debug_path}/frames/*.png")
    texts = []
    crop_images = []
    print("=== Crop images ===")
    for i, img in enumerate(tqdm(images)):
        im=cv2.imread(img)
        crop_path = _crop_image(im, debug_path / "crop", f"crop_{i:03d}")
        crop_images.append(crop_path)
    crop_images = glob.glob(f"{debug_path}/crop/*.png")
    gray_images = []
    print("=== Convert to gray ===")
    for i, crop in enumerate(tqdm(crop_images)):
        im_gray = to_gray(crop, debug_path / "gray", f"crop_gray_{i:03d}")
        gray_images.append(im_gray)

    gray_images = glob.glob(f"{debug_path}/gray/*.png")

    if args.img_scale == "gray":
        targer_images = gray_images
    else:
        targer_images = crop_images
    print("=== OCR images ===")
    for i, gray in enumerate(tqdm(targer_images)):
        text = ocr_with_winrt(gray)
        text = text.replace(" ", "")
        text = text.replace("①", "")
        text = text.replace("↓", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("（", "")
        text = text.replace("）", "")
        text = text.replace("@", "")
        text = text.replace("、", ",")
        text = text.replace("，", ",")
        text = text.replace("30/30", "")
        text = text.replace("人", " 人")
        text = text.replace("ファン数", "ファン数 ")
        f = open(debug_path / f"text/text-{i:03d}.txt", 'w', encoding='utf-8')
        f.write(text)
        f.close()
        texts.append(text)
    f = open('input/menberList.txt', 'r', encoding='utf-8')
    menber_list = [line.strip() for line in f]
    f = open('input/menberReplace.json', 'r', encoding='utf-8')
    menber_replace = json.load(f)
    f.close()
    print("===Text Converting...===")
    for i, text in enumerate(texts):
        for menber in menber_list:
            if menber in menber_replace.keys():
                for repname in menber_replace[menber]:
                    texts[i] = texts[i].replace(repname, menber)
            if menber in text:
                texts[i] = texts[i].replace(menber, f"\n{menber} ")
    texts = "\n".join(texts).split("\n")
    join_text = "\n".join(texts)
    f = open(debug_path / "text/output.txt", 'w', encoding='utf-8')
    f.write(join_text)
    f.close()
    print("=== Extracting fan count... ===")
    fan_counts = {}
    fans = []
    for i, menber in enumerate(tqdm(menber_list)):
        fan_count = get_fan_count(texts, menber, fans)
        if fan_count is not None:
            fan_counts[menber] = fan_count
            fans.append(fan_count)
        else:
            fan_counts[menber] = 0

    
    json.dump(fan_counts, open('output/output.json', 'w', encoding='UTF-8'), ensure_ascii=False, indent=4)

    if not args.debug:
        shutil.rmtree(debug_path, ignore_errors=True)
    
    print("=== OCR TEXT ===")
    # print(text)