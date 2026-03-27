#!/usr/bin/env python3
import time
import subprocess
import shlex
from pathlib import Path

import requests
from evdev import InputDevice, list_devices, ecodes

KEYBOARD_NAME = "ESP32_KEYB"
PHOTO_DIR = Path.home() / "Pictures"
PHOTO_DIR.mkdir(exist_ok=True)

SERVER_URL = "http://10.23.194.85:8000/detect"

pressed = set()
last_trigger = 0.0
cooldown_s = 1.5

def speak(text: str) -> None:
    text = str(text).strip()
    if not text:
        return

    safe = shlex.quote(text)
    cmd = f"espeak-ng {safe} --stdout | aplay"
    subprocess.run(cmd, shell=True, check=False)

def interpret_server_response(data: dict) -> str:
    success = data.get("success", False)

    if success:
        closest_object = data.get("closest_object")
        if closest_object:
            return str(closest_object)

    error = data.get("error", "")
    if error == "ball_not_detected":
        return "Unsuccessful. Try pointing the camera down."

    return "Unsuccessful. Please try again."

def find_keyboard(name: str) -> InputDevice:
    while True:
        for path in list_devices():
            dev = InputDevice(path)
            if name.lower() in dev.name.lower():
                print(f"Using {dev.path} : {dev.name}", flush=True)
                return dev
        print(f"Waiting for input device containing '{name}'...", flush=True)
        time.sleep(2)


def take_photo() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = PHOTO_DIR / f"capture_{ts}.jpg"

    cmd = [
        "rpicam-still",
        "--output", str(out),
        "--timeout", "1",
        "--immediate",
        "--nopreview",
        "--width", "1296",
        "--height", "972",
    ]

    subprocess.run(cmd, check=True)
    print(f"Saved {out}", flush=True)
    return out


def send_photo_http(path: Path) -> None:
    with open(path, "rb") as f:
        files = {
            "file": (path.name, f, "image/jpeg")
        }
        response = requests.post(SERVER_URL, files=files, timeout=60)

    print(f"Upload status: {response.status_code}", flush=True)
    response.raise_for_status()

    data = response.json()
    print(data, flush=True)

    message = interpret_server_response(data)
    print(f"Speaking: {message}", flush=True)
    speak(message)

    return True


def maybe_trigger():
    global last_trigger
    now = time.time()

    combo = {
        ecodes.KEY_LEFTCTRL,
        ecodes.KEY_LEFTALT,
        ecodes.KEY_P,
    }

    if combo.issubset(pressed) and now - last_trigger > cooldown_s:
        last_trigger = now
        try:
            photo = take_photo()
            send_photo_http(photo)

            if photo.exists():
                photo.unlink()
                print(f"Deleted {photo}", flush=True)

        except Exception as e:
            print(f"Capture/upload failed: {e}", flush=True)
            if photo is not None and photo.exists():
                print(f"Keeping file for retry/debug: {photo}", flush=True)


def main():
    while True:
        dev = find_keyboard(KEYBOARD_NAME)

        try:
            for event in dev.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue

                code = event.code
                value = event.value  # 1=down, 0=up, 2=hold

                if value == 1:
                    pressed.add(code)
                    maybe_trigger()
                elif value == 0:
                    pressed.discard(code)

        except OSError:
            print("Keyboard disconnected. Re-scanning...", flush=True)
            pressed.clear()
            time.sleep(1)


if __name__ == "__main__":
    main()
