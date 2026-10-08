"""
NeuroSpeech Bridge - Stage 7 Speech Output
=========================================
This module adds text-to-speech output using edge-tts.

How edge-tts works:
- edge-tts uses Microsoft Azure Neural voices over the network.
- It converts plain text into a WAV audio stream.
- The stream can then be played with a local audio player.

Why asynchronous speech is needed:
- Speech synthesis can take a short but noticeable amount of time.
- If it runs in the main webcam loop, the loop may freeze or become delayed.
- Running speech in a background worker keeps the camera and classification flow
  responsive.

How cooldown prevents spam:
- The same detected phrase may be repeated many times in a row.
- A cooldown period prevents the system from speaking the same phrase again
  immediately and making the output feel noisy or repetitive.

Important note:
- This module only adds speech output functionality.
- It does not modify DTW or calibration logic.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import edge_tts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VOICE = "en-IN-NeerjaNeural"
COOLDOWN_SECONDS = 3.0
AUDIO_PLAYER = "powershell"


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_last_spoken_phrase: Optional[str] = None
_last_spoken_time: float = 0.0
_is_muted = False
_speech_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _now_timestamp() -> str:
    """Return a readable timestamp for console output."""
    return datetime.now().strftime("%H:%M:%S")


async def _speak_async(text: str, output_path: Path) -> None:
    """Generate speech audio asynchronously using edge-tts."""
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(str(output_path))


def _play_audio_file(audio_path: Path) -> None:
    """Play the generated WAV file using the system audio player."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file was not created: {audio_path}")

    try:
        subprocess.run(
            [AUDIO_PLAYER, "-c", "Add-Type -AssemblyName presentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open((Resolve-Path '" + str(audio_path).replace("'", "''") + "').Path); $player.Play(); Start-Sleep -Seconds 5"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Audio playback failed: {exc}") from exc


def speak(text: str, force: bool = False) -> bool:
    """
    Speak a phrase using edge-tts and play it asynchronously.

    Parameters:
        text: The text to convert to speech.
        force: If True, bypass the cooldown check.

    Returns:
        True if the phrase was spoken, False if suppressed by cooldown or mute.
    """
    global _last_spoken_phrase, _last_spoken_time, _is_muted

    if not text or not text.strip():
        print("TTS status: empty phrase, skipping")
        return False

    phrase = text.strip()

    with _speech_lock:
        if _is_muted:
            print(f"TTS status: muted, skipped -> {phrase}")
            return False

        now = time.time()
        if not force and _last_spoken_phrase == phrase and (now - _last_spoken_time) < COOLDOWN_SECONDS:
            print(f"TTS status: cooldown active, skipped -> {phrase}")
            return False

        _last_spoken_phrase = phrase
        _last_spoken_time = now

    print(f"Spoken phrase: {phrase}")
    print(f"Timestamp: {_now_timestamp()}")
    print("TTS status: generating speech")

    def worker() -> None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = Path(tmp.name)

            try:
                asyncio.run(_speak_async(phrase, output_path))
                if output_path.exists():
                    print("TTS status: playback started")
                    _play_audio_file(output_path)
                else:
                    raise FileNotFoundError("Generated audio file not found")
            except Exception as exc:
                print(f"TTS status: error -> {exc}")
            finally:
                if output_path.exists():
                    try:
                        output_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as exc:
            print(f"TTS status: worker error -> {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return True


def mute() -> None:
    """Mute speech output."""
    global _is_muted
    _is_muted = True
    print("TTS status: muted")


def unmute() -> None:
    """Unmute speech output."""
    global _is_muted
    _is_muted = False
    print("TTS status: unmuted")


def toggle_mute() -> bool:
    """Toggle mute state and return the new state."""
    if _is_muted:
        unmute()
        return False
    mute()
    return True


def demo_mode() -> None:
    """Run a small terminal-based demo for text-to-speech."""
    print("\nTTS demo mode")
    print("Type a phrase and press Enter. Type 'quit' to exit.")

    while True:
        try:
            text = input("Phrase > ").strip()
        except KeyboardInterrupt:
            print("\nDemo mode exited")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("Demo mode exited")
            break

        speak(text, force=True)


if __name__ == "__main__":
    try:
        demo_mode()
    except Exception as exc:
        print(f"TTS error: {exc}")
