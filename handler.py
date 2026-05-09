import runpod
import os
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/cache/nemo-models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["NEMO_CACHE_DIR"] = CACHE_DIR


def prepare_audio(input_path: str, output_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def run_diarization(audio_path: str, num_speakers=None, max_speakers: int = 8) -> list:
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write(json.dumps({
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": num_speakers,
                "rttm_filepath": None,
                "uem_filepath": None,
            }) + "\n")

        cfg = OmegaConf.create({
            "num_workers": 0,
            "sample_rate": 16000,
            "batch_size": 64,
            "device": "cuda",
            "verbose": False,
            "diarizer": {
                "manifest_filepath": manifest_path,
                "out_dir": tmp_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.875,
                        "onset": 0.4,
                        "offset": 0.4,
                        "pad_onset": 0.4,
                        "pad_offset": 0.0,
                        "min_duration_on": 0.2,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.1],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": num_speakers is not None,
                        "max_num_speakers": num_speakers if num_speakers else max_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    },
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephony",
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [0.7],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_eval_settings": [[0.25, True]],
                        "debug_mode": False,
                        "use_clus_as_main": False,
                        "pairwise_infer": True,
                    },
                },
            },
        })

        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        rttm_files = list(Path(tmp_dir).rglob("pred_rttms/*.rttm"))
        if not rttm_files:
            rttm_files = list(Path(tmp_dir).rglob("*.rttm"))

        segments = []
        if rttm_files:
            with open(rttm_files[0]) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        segments.append({
                            "start": round(start, 3),
                            "end": round(start + duration, 3),
                            "speaker": speaker,
                        })

        return sorted(segments, key=lambda x: x["start"])


def handler(job):
    try:
        job_input = job["input"]
        audio_url = job_input.get("audio_url")
        num_speakers = job_input.get("num_speakers")
        max_speakers = job_input.get("max_speakers", 8)

        if not audio_url:
            return {"error": "audio_url is required"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = os.path.join(tmp_dir, "input")
            wav_path = os.path.join(tmp_dir, "audio_16k.wav")

            urllib.request.urlretrieve(audio_url, raw_path)
            prepare_audio(raw_path, wav_path)

            segments = run_diarization(wav_path, num_speakers, max_speakers)
            speakers = list(set(s["speaker"] for s in segments))

            return {
                "status": "success",
                "segments": segments,
                "num_speakers": len(speakers),
                "speakers": speakers,
            }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
