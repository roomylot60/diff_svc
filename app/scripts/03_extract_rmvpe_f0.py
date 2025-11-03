#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Robust Model for Vocal Pitch Estimation in Polyphonic Music(RMVPE): 다성환경 기초주파수 추정 모델
# 프레임 단위로 F0(기초주파수)를 추출 후, 공식 시간축(44.1 kHz / hop=512)에 정렬
class RMVPEModel:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        # TODO: 실제 RMVPE 로더/모델을 import하여 초기화
        # ex) self.model = rmvpe.load_model(ckpt_path, device=device)
        self.device = device
        self.sr_in = 16000  # 내부 추론 SR 가정(필요 시 수정)

    def infer_f0_hz(self, y_16k: np.ndarray, hop_16k: int,
                    fmin: float, fmax: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        입력: 16kHz 모노 파형(y_16k), hop_16k 샘플 간격
        출력: f0_hz[T_in], conf[T_in](없으면 None)
        """
        # TODO: 실제 RMVPE 추론 호출로 대체
        # 여기서는 데모로 librosa.yin을 사용(정확도 < RMVPE)
        # 주의: yin은 프레임 단위를 n_fft/윈도에 따르므로 추후 interp로 정렬
        f0 = librosa.yin(y_16k, fmin=fmin, fmax=fmax, sr=self.sr_in,
                         frame_length=2048, hop_length=hop_16k)
        conf = None
        return f0.astype(np.float32), conf

def compute_num_frames(n_samples: int, hop: int, center: bool = False, win_length: Optional[int] = None) -> int:
    """
    input:
        n_samples: 오디오 총 샘플 수(예: 44.1kHz 기준)
        hop: 목표 hop 길이(샘플), 기본 512
        center: STFT center 패딩 여부(파이프라인 통일을 위해 False 권장)
        win_length: 윈도 길이(필수 아님)
    output:
        T_hop: 목표 시간축에서의 프레임 수(int)
    description:
        hop 기준 목표 프레임 수(T_hop)를 계산.
        RMVPE 내부 윈도와 무관하게, '공식 시간축'은 sr/hop로 정의.
    """
    if not center or win_length is None:
        # center=False 기준의 안전한 정의
        return int(np.floor((n_samples - 1) / hop) + 1)
    # center=True일 때는 패딩으로 1~2프레임 증가 가능하지만,
    # 파이프라인을 center=False로 통일하는 것을 권장.
    return int(np.floor((n_samples - win_length) / hop) + 1)

def seconds_from_indices(indices: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """
    input:
        indices: 프레임 인덱스 배열(0..T-1)
        sr, hop: 시간축 정의
    output:
        각 프레임 인덱스에 대응하는 초 단위 시간 배열
    description:
        “프레임 번호 → 절대 시간(초)” 변환 유틸
    """
    return (indices.astype(np.float64) * hop) / float(sr)

def align_series_timebase(x_in: np.ndarray,
                          sr_in: int, hop_in: int,
                          sr_out: int, hop_out: int,
                          n_out: int,
                          fill_value: float = 0.0) -> np.ndarray:
    """
    input:
        x_in: 원 시계열(길이 T_in) — 보통 RMVPE 원 출력 f0
        sr_in, hop_in: 원 시간축 정의(예: 16k/≈10~12ms)
        sr_out, hop_out: 목표 시간축(여기선 44.1k/512)
        n_out: 목표 길이(=T_hop)
        fill_value: 외삽/결측 보정 값
    output:
        x_out: 목표 시간축으로 선형 보간 정렬된 시계열(길이 = n_out)
    description:
        시계열 x_in[T_in] (sr_in/hop_in 기준)을
        목표 축(sr_out/hop_out)으로 선형 보간 정렬 후 길이를 n_out으로 맞춤
    """
    T_in = len(x_in)
    if T_in == 0:
        return np.full((n_out,), fill_value, dtype=np.float32)

    t_in = seconds_from_indices(np.arange(T_in), sr_in, hop_in)
    t_out = seconds_from_indices(np.arange(n_out), sr_out, hop_out)

    # np.interp는 x 증가 가정, 외삽은 양끝 값으로 처리 → 이후 무성 처리로 보완
    x_out = np.interp(t_out, t_in, x_in).astype(np.float32)
    return x_out

def hz_to_midi(f0_hz: np.ndarray, invalid_to_nan: bool = True) -> np.ndarray:
    """
    input:
        f0_hz: Hz 단위 F0 시퀀스
        invalid_to_nan: 0 이하를 NaN 처리 여부
    output:
        f0_midi: MIDI 단위(69 + 12*log2(f/440)) 시퀀스
    description:
        피치 쉬프트, 키 분석 등 후속 활용을 위한 변환
    """
    f0 = f0_hz.copy().astype(np.float64)
    if invalid_to_nan:
        f0[f0 <= 0] = np.nan
    midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
    return midi.astype(np.float32)

def median_denoise(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=k).astype(np.float32)

def load_rmvpe_model(ckpt_path: str, device: str) -> RMVPEModel:
    """
    input:
        ckpt_path: RMVPE 체크포인트 경로
        device: "cuda" 또는 "cpu"
    output:
        RMVPEModel 인스턴스(추론용, eval 상태)
    description:
        RMVPE 추론 객체를 초기화합니다. (내부 추론 샘플레이트 sr_in 예: 16kHz)
    """
    return RMVPEModel(ckpt_path, device=device)

def extract_f0_rmvpe(y_44k: np.ndarray,
                     sr_44k: int,
                     hop_out: int,
                     model: RMVPEModel,
                     fmin: float,
                     fmax: float,
                     voicing_thresh: float,
                     t_out: int,
                     apply_median_k: int = 3) -> Dict[str, np.ndarray]:
    """
    44.1k 파형을 RMVPE 요구 SR(예: 16k)로 리샘플 → f0 추정 → 공식 축(44.1k/hop_out)으로 정렬
    """

    # 1) RMVPE 입력 SR로 리샘플 (고품질)
    y_16k = librosa.resample(y_44k, orig_sr=sr_44k, target_sr=model.sr_in, res_type="kaiser_best").astype(np.float32)

    # 2) RMVPE 프레임 간격 설정(16k 기준 10ms ≈ hop_in=160 권장)
    hop_in = int(round(hop_out * model.sr_in / sr_44k))  # 512 * 16000 / 44100 ≈ 186
    hop_in = max(1, hop_in)

    # 3) f0 추정 (Hz)
    f0_raw_hz, conf_raw = model.infer_f0_hz(y_16k, hop_in, fmin, fmax)

    # 4) 공식 시간축(44.1k/hop_out)으로 정렬
    f0_hz = align_series_timebase(f0_raw_hz, model.sr_in, hop_in, sr_44k, hop_out, t_out, fill_value=0.0)

    # 5) 대역 밖 제거 + 무성 처리
    #   (RMVPE 출력에 노이즈가 남는 경우를 대비)
    uv_mask = (f0_hz >= fmin) & (f0_hz <= fmax)
    f0_hz[~uv_mask] = 0.0

    # 6) 선택적 스파이크 제거(메디안)
    if apply_median_k and apply_median_k >= 3:
        # 무성(0)은 필터로 퍼지지 않게 별도 마스크 처리
        voiced_idx = np.where(uv_mask)[0]
        if len(voiced_idx) > 0:
            tmp = f0_hz.copy()
            tmp_voiced = tmp.copy()
            tmp_voiced[~uv_mask] = np.nan
            # NaN-safe median: 양 끝단 NaN은 유지 → 이후 0 처리
            tmp_voiced = np.where(np.isnan(tmp_voiced), 0.0, median_denoise(np.nan_to_num(tmp_voiced), k=apply_median_k))
            # 필터 후에도 대역 제한 재적용
            tmp_voiced[(tmp_voiced < fmin) | (tmp_voiced > fmax)] = 0.0
            f0_hz = tmp_voiced.astype(np.float32)
            uv_mask = (f0_hz > 0)

    return {
        "f0_hz": f0_hz.astype(np.float32),
        "uv_mask": uv_mask.astype(bool),
        "conf": None if conf_raw is None else conf_raw.astype(np.float32)
    }

def process_file(wav_path: str,
                 out_dir: str,
                 model: RMVPEModel,
                 hop_out: int,
                 fmin: float,
                 fmax: float,
                 voicing_thresh: float,
                 save_midi: bool,
                 median_k: int) -> None:
    base = os.path.splitext(os.path.basename(wav_path))[0]

    # 1) 로드 (44.1k 모노 전제. 아닐 경우 44.1k로 재샘플 권장)
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 44100:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=44100, res_type="kaiser_best")
        sr = 44100
    y = y.astype(np.float32)

    # 2) 목표 프레임 수(t_out) 산정
    t_out = compute_num_frames(len(y), hop_out, center=False)

    # 3) f0 추출 + 정렬
    result = extract_f0_rmvpe(
        y_44k=y,
        sr_44k=sr,
        hop_out=hop_out,
        model=model,
        fmin=fmin,
        fmax=fmax,
        voicing_thresh=voicing_thresh,
        t_out=t_out,
        apply_median_k=median_k
    )
    f0_hz = result["f0_hz"]
    uv_mask = result["uv_mask"]

    # 4) 길이 보정(안전)
    if len(f0_hz) != t_out:
        if len(f0_hz) > t_out:
            f0_hz = f0_hz[:t_out]
            uv_mask = uv_mask[:t_out]
        else:
            pad_n = t_out - len(f0_hz)
            f0_hz = np.pad(f0_hz, (0, pad_n), constant_values=0.0)
            uv_mask = np.pad(uv_mask, (0, pad_n), constant_values=False)

    # 5) 저장
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{base}_f0_hz.npy"), f0_hz.astype(np.float32))
    np.save(os.path.join(out_dir, f"{base}_uv_mask.npy"), uv_mask.astype(bool))

    if save_midi:
        f0_midi = hz_to_midi(f0_hz, invalid_to_nan=True)
        np.save(os.path.join(out_dir, f"{base}_f0_midi.npy"), f0_midi.astype(np.float32))

def main():
    ap = argparse.ArgumentParser(description="RMVPE-based F0 extractor (44.1k/hop512 aligned)")
    ap.add_argument("--input_dir", type=str, required=True, help="resampled_audio/ (44.1k wav)")
    ap.add_argument("--output_dir", type=str, required=True, help="features/f0/")
    ap.add_argument("--rmvpe_ckpt", type=str, required=True, help="RMVPE checkpoint path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hop", type=int, default=512, help="hop length on 44.1k timeline")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=1100.0)
    ap.add_argument("--voicing_thresh", type=float, default=0.5)
    ap.add_argument("--median_k", type=int, default=3, help="odd kernel size; 0/1 to disable")
    ap.add_argument("--save_midi", action="store_true")
    args = ap.parse_args()

    model = load_rmvpe_model(args.rmvpe_ckpt, args.device)

    wavs = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    os.makedirs(args.output_dir, exist_ok=True)

    for i, wp in enumerate(wavs, 1):
        process_file(
            wav_path=wp,
            out_dir=args.output_dir,
            model=model,
            hop_out=args.hop,
            fmin=args.fmin,
            fmax=args.fmax,
            voicing_thresh=args.voicing_thresh,
            save_midi=args.save_midi,
            median_k=args.median_k
        )
        print(f"[{i}/{len(wavs)}] done: {os.path.basename(wp)}")


if __name__ == "__main__":
    main()
