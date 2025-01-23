import os
import glob
import random
import soundfile
from tqdm import tqdm

def mapping_emotion(emotion):
    if emotion == "NEU":
        return "<|NEUTRAL|>"
    if emotion == "neutral":
        return "<|NEUTRAL|>"
    elif emotion == "sad":
        return "<|HAPPY|>"
    elif emotion == "angry":
        return "<|ANGRY|>"
    elif emotion == "happy":
        return "<|HAPPY|>"
    elif emotion == "NEG":
        return "<|SAD|>"
    elif emotion == "POS":
        return "<|HAPPY|>"
    else:
        raise ValueError(f"Unknown emotion: {emotion}")

def process_datasets(input_dir: str, output_dir: str):
    """
    Process ASR datasets and generate required files from transcripts.txt in each dataset folder.

    :param input_dir: Path to the folder containing datasets (e.g., data/asr/<datasets>).
    :param output_dir: Path to the folder where output files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file paths
    train_text_path = os.path.join(output_dir, "train_text.txt")
    train_wav_scp_path = os.path.join(output_dir, "train_wav.scp")
    train_text_language_path = os.path.join(output_dir, "train_text_language.txt")
    train_emo_path = os.path.join(output_dir, "train_emo.txt")
    train_event_path = os.path.join(output_dir, "train_event.txt")
    train_path = os.path.join(output_dir, "train.txt")

    # Define the file paths
    valid_text_path = os.path.join(output_dir, "valid_text.txt")
    valid_wav_scp_path = os.path.join(output_dir, "valid_wav.scp")
    valid_text_language_path = os.path.join(output_dir, "valid_text_language.txt")
    valid_emo_path = os.path.join(output_dir, "valid_emo.txt")
    valid_event_path = os.path.join(output_dir, "valid_event.txt")
    valid_path = os.path.join(output_dir, "valid.txt")

    # Initialize files
    with open(train_text_path, "w", encoding="utf-8") as text_file, \
         open(train_wav_scp_path, "w", encoding="utf-8") as wav_scp_file, \
         open(train_text_language_path, "w", encoding="utf-8") as lang_file, \
         open(train_emo_path, "w", encoding="utf-8") as emo_file, \
         open(train_event_path, "w", encoding="utf-8") as event_file, \
         open(train_path, "w", encoding="utf-8") as train_file, \
         open(valid_text_path, "w", encoding="utf-8") as valid_text_file, \
         open(valid_wav_scp_path, "w", encoding="utf-8") as valid_wav_scp_file, \
         open(valid_text_language_path, "w", encoding="utf-8") as valid_lang_file, \
         open(valid_emo_path, "w", encoding="utf-8") as valid_emo_file, \
         open(valid_event_path, "w", encoding="utf-8") as valid_event_file, \
         open(valid_path, "w", encoding="utf-8") as valid_file:

        # Find all transcripts.txt files in the dataset folder
        transcript_files = glob.glob(os.path.join(input_dir, "**", "transcripts.txt"), recursive=True)
        transcript_files += glob.glob(os.path.join(input_dir, "**", "metadata.txt"), recursive=True)

        # Iterate through each transcripts.txt file
        total = 0
        for i, transcript_file in enumerate(transcript_files):
            with open(transcript_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.seed(42)
                random.shuffle(lines)
                line_idx = 0
                for line in tqdm(lines, desc=f"Processing [{i+1}/{len(transcript_files)}] {transcript_file.split('/')[-2]}"):
                    line_idx += 1
                    parts = line.strip().split("|")
                    if "transcripts.txt" in transcript_file:
                        if "emotion" in transcript_file:
                            if "emotion_vlsp_2022" in transcript_file:
                                wav_path = parts[0]
                                text = parts[1]
                                duration = parts[2]
                                emotion = parts[3]
                            else:
                                wav_path = parts[0]
                                emotion = parts[1]
                                duration = parts[2]
                                text = parts[3]
                                emotion = mapping_emotion(emotion)
                        else:
                            wav_path = parts[0]
                            text = parts[1]
                            duration = parts[2]
                            emotion = "<|NEUTRAL|>"

                        if "ctv" in transcript_file:
                            if '111' in text or '000' in text:
                                continue
                            text = text.strip().lower()
                            text = text.replace(' ck', '')
                            text = text.replace(' unk', '')
                        
                        c = 0
                        while not os.path.exists(wav_path):
                            if "/" not in wav_path:
                                wav_path = transcript_file.replace("transcripts.txt", "wavs/") + wav_path
                            if "/data/raw/train" in wav_path:
                                wav_path = wav_path.replace("/data/raw/train", "/home/andrew/data/ASR")
                            if "/home/andrew/data/asr/" in wav_path:
                                wav_path = wav_path.replace("/home/andrew/data/asr/", "/home/andrew/data/ASR/")
                            if "/home/andrew/data/ASR" not in wav_path:
                                wav_path = "/home/andrew/data/ASR" + wav_path.strip(".")
                            c += 1
                            if c > 5:
                                break
                    elif "metadata.txt" in transcript_file:
                        wav_path = parts[1]
                        text = parts[2].strip()
                        emotion = "<|NEUTRAL|>"

                    text = text.strip()
                    utt_id = f'{transcript_file.split("/")[-2]}_{os.path.basename(wav_path).split(".")[0]}'

                    if not os.path.exists(wav_path):
                        print(f"[{transcript_file}] File not found: {wav_path}")
                        continue

                    if not len(text.split()):
                        print(f"Empty text: {utt_id}")
                        continue

                    try:
                        if os.path.getsize(wav_path) < 1024*10:
                            print(f"File size too small: {wav_path}")
                            continue
                    except Exception as e:
                        print(f"Error reading audio file: {wav_path} {e}")
                        continue

                    # if duration < 0.3 or duration > 30.0:
                    #     print(f"Warning: duration {duration}s out of range [0.5, 20.0], skipping...")
                    #     continue
                    
                    # Write data to files
                    if line_idx > 0.995 * len(lines):
                        valid_text_file.write(f"{utt_id} {text}\n")
                        valid_wav_scp_file.write(f"{utt_id} {wav_path}\n")
                        valid_lang_file.write(f"{utt_id} <|vi|>\n")
                        valid_emo_file.write(f"{utt_id} {emotion}\n")
                        valid_event_file.write(f"{utt_id} <|Speech|>\n")
                        valid_file.write(f"{wav_path}|{text}\n")
                    else:
                        text_file.write(f"{utt_id} {text}\n")
                        wav_scp_file.write(f"{utt_id} {wav_path}\n")
                        lang_file.write(f"{utt_id} <|vi|>\n")
                        emo_file.write(f"{utt_id} {emotion}\n")
                        event_file.write(f"{utt_id} <|Speech|>\n")
                        train_file.write(f"{wav_path}|{text}\n")
                    
                    total += 1
    print(f"Files generated and saved to: {output_dir} [Total: {total} utterances]")


# Example usage
input_directory = "/home/andrew/data"
output_directory = "data"
process_datasets(input_directory, output_directory)