import os
import glob
from tqdm import tqdm

def process_datasets(input_dir: str, output_dir: str):
    """
    Process ASR datasets and generate required files from transcripts.txt in each dataset folder.

    :param input_dir: Path to the folder containing datasets (e.g., data/asr/<datasets>).
    :param output_dir: Path to the folder where output files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    transcript_files = glob.glob(os.path.join(input_dir, "**", "transcripts_lower.txt"), recursive=True)
    for i, transcript_file in enumerate(transcript_files):
        if "test" not in transcript_file:
            continue
        with open(transcript_file, "r", encoding="utf-8") as f:
            test_set = transcript_file.split("/")[-2]
            test_dir = f"{output_dir}/{test_set}"
            os.makedirs(test_dir, exist_ok=True)

            test_text_path = os.path.join(test_dir, "text")
            test_wav_scp_path = os.path.join(test_dir, "wav.scp")

            with open(test_text_path, "w", encoding="utf-8") as text_file, \
                open(test_wav_scp_path, "w", encoding="utf-8") as wav_scp_file:
                    lines = f.readlines()
                    for line in tqdm(lines, desc=f"Processing [{i+1}/{len(transcript_files)}] {transcript_file.split('/')[-2]}"):
                        parts = line.strip().split("|")
                        wav_path = parts[0]
                        text = parts[1].strip(".?!").strip()
                        utt_id = f'{transcript_file.split("/")[-2]}_{os.path.basename(wav_path).split(".")[0]}'
                        text_file.write(f"{utt_id} {text}\n")
                        wav_scp_file.write(f"{utt_id} {wav_path}\n")
        print(f"Files generated and saved to: {test_dir}")

input_directory = "/home/andrew/data"
output_directory = "/home/andrew/wenet/examples/vietasr/s0/data_asr_tts_new_lower"
process_datasets(input_directory, output_directory)
