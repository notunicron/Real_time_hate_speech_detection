import os
import sys
import numpy as np
import torch
import sounddevice as sd
import keyboard # Use 'pip install keyboard'
from transformers import pipeline
from colorama import init, Fore, Style, deinit
import time
import queue
import threading
import re # Import regex for more flexible cleaning

# --- Configuration ---
HATE_MODEL_PATH = "./hatexplain_word_level_model_embedded_rationales" # Path to your trained model
ASR_MODEL = "openai/whisper-base.en"
SAMPLE_RATE = 16000
# --- MODIFIED: Slightly larger chunk might help reduce edge artifacts ---
CHUNK_DURATION_S = 3 # Process audio in chunks of this duration (seconds)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
STOP_KEY = 'q'
# --- Threshold for filtering out very short/likely noise transcriptions ---
MIN_TRANSCRIPTION_LENGTH = 2 # Ignore transcriptions shorter than this

# --- Check for GPU (remains the same) ---
if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(device)
    print(f"GPU found: {device_name}. Using GPU (device: {device}).")
    torch_device = torch.device("cuda:0")
else:
    device = -1
    print("GPU not found or CUDA not configured. Using CPU.")
    torch_device = torch.device("cpu")

# --- Load Models (remains the same) ---
print("Loading models...")
try:
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        device=device
    )
    print(f"ASR model ({ASR_MODEL}) loaded.")

    if not os.path.exists(HATE_MODEL_PATH):
        print(f"Error: Trained model not found at {HATE_MODEL_PATH}")
        sys.exit(1)

    word_hate_pipe = pipeline(
        "token-classification",
        model=HATE_MODEL_PATH,
        tokenizer=HATE_MODEL_PATH,
        aggregation_strategy="simple",
        device=device
    )
    print(f"Hate speech model ({HATE_MODEL_PATH}) loaded.")

except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# --- Audio Input Setup (remains the same) ---
audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def capture_audio():
    try:
        with sd.InputStream(callback=audio_callback,
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype='float32',
                            blocksize=CHUNK_SAMPLES):
            print(f"\n--- Listening (Sample Rate: {SAMPLE_RATE}Hz, Chunk Size: {CHUNK_DURATION_S}s) ---")
            print(f"--- Press '{STOP_KEY}' to stop ---")
            stop_event.wait()
            print("\n--- Stopping audio capture ---")
    except Exception as e:
        print(f"\nError during audio capture: {e}")
        stop_event.set()

# --- Main Processing Loop (MODIFIED) ---
def process_and_print():
    """Processes audio, transcribes, analyzes, filters noise, and prints."""
    init(autoreset=True) # Initialize colorama
    # We don't need full_transcription unless we want to store everything

    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
            audio_chunk = audio_chunk.flatten()

            # 1. Transcribe Audio
            asr_result = asr_pipe(audio_chunk.copy()) # No language needed for .en
            raw_text = asr_result["text"].strip() if asr_result and "text" in asr_result else ""

            # --- FILTERING ---
            # Remove leading/trailing spaces and normalize common spurious outputs
            text_to_process = raw_text.strip().lower()

            # Filter common Whisper hallucinations during silence/noise
            common_hallucinations = {"you", "thank you.", "thanks for watching!"} # Add others if observed
            if not text_to_process or \
               text_to_process in common_hallucinations or \
               len(raw_text.strip()) < MIN_TRANSCRIPTION_LENGTH: # Filter very short results
                continue # Skip this chunk if it's empty, just "you", or too short

            # Use the original casing for analysis and printing
            text = raw_text.strip()

            # 2. Analyze for Hate Speech
            hate_results = word_hate_pipe(text)

            # 3. Highlight and Print
            highlighted_text = ""
            hate_words = {res['word'].lower() for res in hate_results if res['entity_group'] == 'HATE'}

            # Use regex to split text into words and punctuation, preserving spaces somewhat
            # This handles punctuation better than simple split()
            tokens = re.findall(r'\w+|[^\w\s]|\s+', text)

            for token in tokens:
                # Check if the token is a word
                if token.isalnum(): # Check if it's alphanumeric (a word)
                    clean_word = token.lower()
                    if clean_word in hate_words:
                        highlighted_text += Fore.RED + Style.BRIGHT + token + Style.RESET_ALL
                    else:
                        highlighted_text += token
                else:
                    # Keep punctuation and spaces as they are
                    highlighted_text += token

            # --- PRINTING (MODIFIED) ---
            # Print each new chunk on its own line without overwriting
            print(f"{highlighted_text.strip()}") # No \r, default end="\n"

        except queue.Empty:
            time.sleep(0.1)
            continue
        except Exception as e:
            print(f"\nError during processing loop: {e}")
            # Optionally print traceback for debugging:
            # import traceback
            # traceback.print_exc()
            time.sleep(0.5)

    deinit()
    print("\n--- Processing finished ---")


# --- Stop Mechanism (remains the same) ---
def check_stop_key():
    keyboard.wait(STOP_KEY)
    print(f"\n'{STOP_KEY}' pressed.")
    stop_event.set()

# --- Start Threads (remains the same) ---
if __name__ == "__main__":
    print("Initializing...")

    capture_thread = threading.Thread(target=capture_audio, daemon=True)
    capture_thread.start()

    stop_key_thread = threading.Thread(target=check_stop_key, daemon=True)
    stop_key_thread.start()

    time.sleep(1)
    process_and_print()

    time.sleep(0.5)
    print("Exiting script.")