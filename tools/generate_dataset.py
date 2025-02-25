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
    os.makedirs(output_dir + "/train", exist_ok=True)
    os.makedirs(output_dir + "/dev", exist_ok=True)

    # Define the file paths
    train_dir = output_dir + "/train"
    valid_dir = output_dir + "/dev"
    
    train_text_path = os.path.join(train_dir, "text")
    train_wav_scp_path = os.path.join(train_dir, "wav.scp")
    # train_text_language_path = os.path.join(train_dir, "train_text_language.txt")
    # train_emo_path = os.path.join(train_dir, "train_emo.txt")
    # train_event_path = os.path.join(train_dir, "train_event.txt")
    train_path = os.path.join(train_dir, "train.txt")

    # Define the file paths
    valid_text_path = os.path.join(valid_dir, "text")
    valid_wav_scp_path = os.path.join(valid_dir, "wav.scp")
    # valid_text_language_path = os.path.join(valid_dir, "valid_text_language.txt")
    # valid_emo_path = os.path.join(valid_dir, "valid_emo.txt")
    # valid_event_path = os.path.join(valid_dir, "valid_event.txt")
    valid_path = os.path.join(valid_dir, "valid.txt")

    # Initialize files
        #  open(train_text_language_path, "w", encoding="utf-8") as lang_file, \
        #  open(train_emo_path, "w", encoding="utf-8") as emo_file, \
        #  open(train_event_path, "w", encoding="utf-8") as event_file, \
        #  open(valid_text_language_path, "w", encoding="utf-8") as valid_lang_file, \
        #  open(valid_emo_path, "w", encoding="utf-8") as valid_emo_file, \
        #  open(valid_event_path, "w", encoding="utf-8") as valid_event_file, \

    with open(train_text_path, "w", encoding="utf-8") as text_file, \
         open(train_wav_scp_path, "w", encoding="utf-8") as wav_scp_file, \
         open(train_path, "w", encoding="utf-8") as train_file, \
         open(valid_text_path, "w", encoding="utf-8") as valid_text_file, \
         open(valid_wav_scp_path, "w", encoding="utf-8") as valid_wav_scp_file, \
         open(valid_path, "w", encoding="utf-8") as valid_file:

        # Find all transcripts.txt files in the dataset folder
        transcript_files = glob.glob(os.path.join(input_dir, "**", "transcripts_new.txt"), recursive=True)
        transcript_files += glob.glob(os.path.join(input_dir, "**", "metadata_new.txt"), recursive=True)
        print(transcript_files)
        # Iterate through each transcripts.txt file
        total = 0
        for i, transcript_file in enumerate(transcript_files):
            if "vivos_test" not in transcript_file \
            and "vivoice" not in transcript_file \
            and "MSR-86K" not in transcript_file\
            and "viet_bud500/test" not in transcript_file:
                continue
            with open(transcript_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.seed(42)
                random.shuffle(lines)
                line_idx = 0
                for line in tqdm(lines, desc=f"Processing [{i+1}/{len(transcript_files)}] {transcript_file.split('/')[-2]}"):
                    parts = line.strip().split("|")
                    wav_path = parts[0]
                    text = parts[1].strip()
                    duration = float(parts[2])
                    # emotion = parts[3]

                    if duration > 8 or len(text.split())  > 20:
                        continue

                    utt_id = f'{transcript_file.split("/")[-2]}_{os.path.basename(wav_path).split(".")[0]}'

                    # Write data to files
                    # if line_idx > 0.995 * len(lines):
                        # valid_text_file.write(f"{utt_id} {text}\n")
                        # valid_wav_scp_file.write(f"{utt_id} {wav_path}\n")
                        # valid_lang_file.write(f"{utt_id} <|vi|>\n")
                        # valid_emo_file.write(f"{utt_id} {emotion}\n")
                        # valid_event_file.write(f"{utt_id} <|Speech|>\n")
                        # valid_file.write(f"{wav_path}|{text}\n")
                    # else:
                        # text_file.write(f"{utt_id} {text}\n")
                        # wav_scp_file.write(f"{utt_id} {wav_path}\n")
                        # lang_file.write(f"{utt_id} <|vi|>\n")
                        # emo_file.write(f"{utt_id} {emotion}\n")
                        # event_file.write(f"{utt_id} <|Speech|>\n")
                    line_idx += 1
                    
                    if "vivos_test" in transcript_file:
                        if line_idx >= 200:
                            break
                    elif "viet_bud500/test" in transcript_file:
                        # if line_idx <= 100000:
                        #     continue
                        if line_idx >= 100:
                            break
                    else:
                        if line_idx < 2000:
                            continue
                        if line_idx >= 2100:
                            break
                    valid_file.write(f"{wav_path}|{text}|{duration}\n")

                    total += 1
    print(f"Files generated and saved to: {output_dir} [Total: {total} utterances]")


# Example usage
input_directory = "/home/andrew/data"
output_directory = "/home/andrew/wenet/data_slt"
process_datasets(input_directory, output_directory)
