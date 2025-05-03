# Real-Time Word-Level Hate Speech Detection

This project implements a system for detecting potentially hateful or offensive words in spoken language in real-time. It uses:

1.  **OpenAI's Whisper:** For high-quality automatic speech recognition (ASR) to transcribe audio captured from a microphone.
2.  **Hugging Face Transformers:** To load and run both the Whisper model and a custom fine-tuned model for word-level hate speech detection.
3.  **HateXplain Dataset:** The fine-tuned model is trained on the HateXplain dataset, specifically using the rationale annotations (word-level explanations) to identify which words contribute to a hateful/offensive classification.
4.  **Real-time Processing:** Listens to the microphone, transcribes chunks of audio, analyzes the transcription, and highlights potentially problematic words directly in the console.

## Features

*   Real-time audio transcription from microphone input.
*   Word-level identification of hate speech/offensive language based on fine-tuned model.
*   Console-based highlighting of detected words.
*   GPU acceleration support via PyTorch (if CUDA is available).
*   Configurable stop key ('q' by default) to end the detection process.
*   Includes the Python script to train the custom word-level detection model.

## Requirements

*   **Python 3.8+**
*   **PyTorch:** With CUDA support for GPU acceleration (optional but recommended). Install from [pytorch.org](https://pytorch.org/).
*   **ffmpeg:** Whisper requires `ffmpeg` to be installed and available in your system's PATH.
    *   **Linux:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS:** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` directory to your system's PATH environment variable.
*   **Permissions:** The `keyboard` library might require administrator/root privileges to listen for global key presses, especially on Linux and macOS. You may need to run the real-time script using `sudo python ...` or configure input device permissions.

## Dataset

This project requires the **HateXplain dataset**, specifically a version where the rationale annotations are **embedded within the main `dataset.json` file**.

## Setup

1.  **Clone the repository (or download the scripts):**
    ```bash
    git clone <https://github.com/notunicron/Real_time_hate_speech_detection.git>
    cd <your-repo-directory>
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    .\.venv\Scripts\activate
    ```
3.  **Install PyTorch:** Follow instructions on [pytorch.org](https://pytorch.org/) for your OS and CUDA version (if applicable).
4.  **Install ffmpeg:** As described in the Requirements section.
5.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Download and place the HateXplain dataset:** As described in the Dataset section (e.g., create `./hatexplain_data/` and put `dataset.json` inside).

## Usage

### 1. Training the Hate Speech Model (Optional - if you want to retrain)

*   Ensure the `dataset.json` is correctly placed (e.g., in `./hatexplain_data/`).
*   Modify configuration parameters (like `MODEL_CHECKPOINT`, `OUTPUT_DIR`, `DATASET_FILE`, epochs, batch size) inside the training script (`train.py`) if needed.
*   Run the training script:
    ```bash
    python train.py
    ```
*   This will fine-tune a transformer model (e.g., `bert-base-uncased`) for token classification using the BIO tagging scheme based on the HateXplain rationales.
*   The trained model (including tokenizer and config files) will be saved to the specified output directory (e.g., `./hatexplain_word_level_model_embedded_rationales/`).

### 2. Running the Real-Time Detection

*   Make sure you have either trained the model (Step 1) or have a pre-trained model directory available.
*   Verify that the `HATE_MODEL_PATH` variable inside the real-time script (`detect.py`) points to the correct model directory.
*   Ensure your microphone is working and selected as the default input device.
*   Run the real-time detection script:
    ```bash
    # On Windows or if keyboard permissions are set up:
    python detect.py

    # On Linux/macOS (if keyboard library requires root):
    sudo python detect.py
    ```
*   The script will load the ASR and hate speech models (this might take time on the first run as models are downloaded).
*   It will then start listening to your microphone.
*   Speak clearly. Transcribed text will appear in the console, with words identified as hate speech/offensive highlighted (typically in red).
*   Press the **'q' key** (or the configured `STOP_KEY`) to stop the script gracefully.

## Model Details

*   **ASR Model:** Uses a pre-trained Whisper model (e.g., `openai/whisper-base.en`). Larger models (`small.en`, `medium.en`) may offer better accuracy at the cost of performance and VRAM.
*   **Hate Speech Model:** A Transformer model (default: `bert-base-uncased`) fine-tuned for Token Classification on the HateXplain dataset. It predicts BIO tags (`B-HATE`, `I-HATE`, `O`) for each word token.

## Limitations

*   **ASR Accuracy:** The accuracy of the hate speech detection is heavily dependent on the accuracy of the Whisper transcription. Background noise, unclear speech, or accents can lead to transcription errors, impacting detection.
*   **Hate Speech Model Performance:** The fine-tuned model is not perfect and will have false positives (flagging harmless words) and false negatives (missing hateful words). Its performance depends on the quality and representativeness of the HateXplain dataset and the fine-tuning process.
*   **Context:** This model performs *word-level* detection. It lacks broader sentence or conversational context, which is often crucial for understanding nuance and intent in hate speech. Sarcasm or reclaimed slurs might be misidentified.
*   **Whisper Artifacts:** Whisper models can sometimes "hallucinate" repetitive phrases (like "you" or "Thank you for watching") during periods of silence or noise. The script includes basic filtering for this, but it might not catch all cases.
*   **Real-Time Performance:** Latency depends on your hardware (CPU/GPU), the chosen Whisper model size, and the audio chunk size. There will be a slight delay between speaking and seeing the highlighted transcription.
*   **Keyboard Library Permissions:** Requiring root/admin privileges for the `keyboard` listener can be inconvenient or a security concern in some environments.
*   **Fixed Chunking:** Processing audio in fixed chunks might awkwardly split words or phrases across chunk boundaries.

## Future Improvements

*   Implement more robust Voice Activity Detection (VAD) to avoid processing silence.
*   Combine word-level detection with sentence-level classification for better context.
*   Experiment with larger ASR and language models.
*   Develop a graphical user interface (GUI) instead of console output.
*   Explore alternative libraries for non-privileged key listening.
*   Implement more sophisticated audio buffering and processing strategies.