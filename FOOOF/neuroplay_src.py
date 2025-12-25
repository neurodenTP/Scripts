import numpy as np
import mne
import pandas as pd
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
# from specparam import SpectralGroupModel
from fooof import FOOOFGroup, FOOOF


def annotate_high_amplitude_per_channel(
    raw: mne.io.BaseRaw,
    threshold: float,
    pre: float = 0.0,
    post: float = 0.0,
    desc: str = "BAD_amp",
) -> mne.Annotations:
    sfreq = raw.info["sfreq"]
    ch_idx = mne.pick_types(raw.info, eeg=True)
    data, times = raw[ch_idx, :]  # (n_ch, n_times)

    all_onsets = []
    all_durations = []
    all_desc = []
    all_ch_names = []

    for i, idx in enumerate(ch_idx):
        ch_name = raw.ch_names[idx]
        ch_data = data[i]

        bad_mask = np.abs(ch_data) > threshold
        if not np.any(bad_mask):
            continue

        bad_idx = np.where(bad_mask)[0]
        splits = np.where(np.diff(bad_idx) > 1)[0] + 1
        clusters = np.split(bad_idx, splits)

        for cl in clusters:
            t_start = times[cl[0]] - pre
            t_stop = times[cl[-1]] + post

            t_start = max(t_start, times[0])
            t_stop = min(t_stop, times[-1])
            if t_stop <= t_start:
                continue

            all_onsets.append(t_start)
            all_durations.append(t_stop - t_start)
            all_desc.append(desc)
            all_ch_names.append([ch_name])

    if not all_onsets:
        return mne.Annotations([], [], [], orig_time=raw.info["meas_date"])

    annot = mne.Annotations(
        onset=all_onsets,
        duration=all_durations,
        description=all_desc,
        ch_names=all_ch_names,
        orig_time=raw.info["meas_date"],
    )
    return annot


def merge_overlapping_annotations(annot: mne.Annotations) -> mne.Annotations:
    """Объединить пересекающиеся/соприкасающиеся аннотации
    с одинаковыми description и ch_names.
    """
    if len(annot) == 0:
        return annot

    onsets = np.asarray(annot.onset, float)
    durations = np.asarray(annot.duration, float)
    desc = np.asarray(annot.description, dtype=object)

    starts = onsets
    stops = onsets + durations

    has_ch_names = getattr(annot, "ch_names", None) is not None
    if has_ch_names:
        ch_names_list = [
            tuple(chs) if chs is not None and len(chs) > 0 else ()
            for chs in annot.ch_names
        ]
    else:
        ch_names_list = [()] * len(annot)

    groups = {}
    for i, (d, chs) in enumerate(zip(desc, ch_names_list)):
        key = (d, chs)
        groups.setdefault(key, []).append(i)

    merged_onsets = []
    merged_durations = []
    merged_desc = []
    merged_ch_names = []

    for (key_desc, key_chs), idxs in groups.items():
        idxs = np.array(idxs, int)
        s = starts[idxs]
        e = stops[idxs]

        order = np.argsort(s)
        s = s[order]
        e = e[order]

        cur_start = s[0]
        cur_end = e[0]

        for st, en in zip(s[1:], e[1:]):
            if st <= cur_end:
                cur_end = max(cur_end, en)
            else:
                merged_onsets.append(cur_start)
                merged_durations.append(cur_end - cur_start)
                merged_desc.append(key_desc)
                merged_ch_names.append(list(key_chs))
                cur_start, cur_end = st, en

        merged_onsets.append(cur_start)
        merged_durations.append(cur_end - cur_start)
        merged_desc.append(key_desc)
        merged_ch_names.append(list(key_chs))

    if not has_ch_names:
        merged = mne.Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=merged_desc,
            orig_time=annot.orig_time,
        )
    else:
        merged = mne.Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=merged_desc,
            ch_names=merged_ch_names,
            orig_time=annot.orig_time,
        )
    return merged


def create_channel_bad_masks(raw, picks, bad_prefix="bad"):
    """Матрица масок (n_picks, n_times): True = артефакт в данном канале/время."""
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    picks = np.atleast_1d(picks)
    ch_names = np.array(raw.ch_names)

    bad_masks = np.zeros((len(picks), n_times), dtype=bool)

    ann = raw.annotations
    has_ch = getattr(ann, "ch_names", None) is not None

    for i_ann, a in enumerate(ann):
        desc = a["description"]
        if not desc.lower().startswith(bad_prefix):
            continue

        onset = float(a["onset"])
        duration = float(a["duration"])
        onset_sample = max(int(np.round(onset * sfreq)), 0)
        end_sample = min(
            onset_sample + int(np.round(duration * sfreq)),
            n_times,
        )
        if end_sample <= onset_sample:
            continue

        if has_ch:
            ann_chs = ann.ch_names[i_ann]
            if ann_chs is None or len(ann_chs) == 0:
                affected = np.ones(len(picks), dtype=bool)
            else:
                ann_chs = set(ann_chs)
                affected = np.array(
                    [ch_names[idx] in ann_chs for idx in picks], dtype=bool
                )
        else:
            affected = np.ones(len(picks), dtype=bool)

        bad_masks[affected, onset_sample:end_sample] = True

    return bad_masks


def compute_band_power_array(data, sfreq, fmin, fmax):
    """PSD Уэлча для 1D массива, интеграл мощности в диапазоне."""
    if data.size == 0:
        return np.nan

    n_per_seg = int(min(len(data), sfreq * 1.0))
    psd, freqs = psd_array_welch(
        data[np.newaxis, :],
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_per_seg=n_per_seg,
        average="mean",
        verbose=False,
    )
    return np.trapz(psd[0], freqs)


def sliding_band_powers_to_df_per_channel(
    raw,
    win_len=20.0,
    step=10.0,
    bands=None,
    min_clean_ratio=0.5,
    bad_prefix="bad",
):
    """DataFrame: 'time' + <chan>_<band> мощности или NaN."""
    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "smr": (12, 15),
        }

    sfreq = raw.info["sfreq"]
    times = raw.times

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = np.array(raw.ch_names)[picks]

    bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)

    win_centers = []
    cur_start = 0.0
    while cur_start + win_len <= times[-1]:
        cur_stop = cur_start + win_len
        win_centers.append((cur_start + cur_stop) / 2.0)
        cur_start += step
    win_centers = np.array(win_centers)

    cols = ["time"]
    for ch in ch_names:
        for band_name in bands.keys():
            cols.append(f"{ch}_{band_name}")
    df = pd.DataFrame(index=np.arange(len(win_centers)), columns=cols, dtype=float)
    df["time"] = win_centers

    for wi, center in enumerate(win_centers):
        start = center - win_len / 2.0
        stop = center + win_len / 2.0

        start_sample = int(np.round(start * sfreq))
        stop_sample = int(np.round(stop * sfreq))
        start_sample = max(start_sample, 0)
        stop_sample = min(stop_sample, raw.n_times)
        if stop_sample <= start_sample:
            continue

        data_win, _ = raw[picks, start_sample:stop_sample]

        for ci, ch_name in enumerate(ch_names):
            window_bad = bad_masks[ci, start_sample:stop_sample]
            clean_samples = ~window_bad
            clean_ratio = clean_samples.sum() / len(clean_samples)
            if clean_ratio < min_clean_ratio:
                continue

            x = data_win[ci, clean_samples]
            if x.size < sfreq * 0.5:
                continue

            for band_name, (fmin, fmax) in bands.items():
                val = compute_band_power_array(x, sfreq, fmin, fmax)
                val = val / (fmax - fmin)
                df.at[wi, f"{ch_name}_{band_name}"] = val

    return df


def plot_annotation(raw):
    sfreq = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    n_t = raw.n_times
    
    bad_mat = np.zeros((n_ch, n_t), dtype=bool)
    ann = raw.annotations
    has_ch = getattr(ann, "ch_names", None) is not None
    
    for i_ann, a in enumerate(ann):
        desc = a["description"]
        if not desc.lower().startswith("bad"):
            continue
    
        onset = float(a["onset"])
        duration = float(a["duration"])
        start = max(int(np.round(onset * sfreq)), 0)
        stop = min(start + int(np.round(duration * sfreq)), n_t)
        if stop <= start:
            continue
    
        if has_ch:
            ann_chs = ann.ch_names[i_ann]
            if ann_chs is None or len(ann_chs) == 0:
                affected_idx = np.arange(n_ch)
            else:
                affected_idx = [ch_names.index(ch) for ch in ann_chs if ch in ch_names]
        else:
            affected_idx = np.arange(n_ch)
    
        bad_mat[affected_idx, start:stop] = True
    
    fig, ax = plt.subplots(figsize=(15, 8))
    img = ax.imshow(
        bad_mat,
        aspect="auto",
        interpolation="nearest",
        extent=[times[0], times[-1], -0.5, n_ch - 0.5],
        cmap="Reds",
        origin="lower",
    )
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Каналы")
    ax.set_title("BAD-аннотации по всем каналам")
    plt.colorbar(img, ax=ax, label="BAD (1) / OK (0)")
    plt.tight_layout()
    
    
def plot_bands_power(df_bands, bands, dir_name=None, resp_name=None):
    """
    df_bands: DataFrame с колонками 'time' и <chan>_<band>.
    bands: dict, ключи = имена ритмов (theta/alpha/...).
    """
    time = df_bands["time"].values
    band_names = list(bands.keys())

    # извлекаем имена каналов из колонок "<chan>_<band>"
    pattern = r"^(.*)_(" + "|".join(band_names) + r")$"
    ch_names = []
    for col in df_bands.columns:
        m = re.match(pattern, col)
        if m:
            ch = m.group(1)
            if ch not in ch_names:
                ch_names.append(ch)

    # --- 1. ритмы по каналам (subplot на каждый канал) ---
    n_channels = len(ch_names)
    n_cols = 2
    n_rows = (n_channels + 1) // 2

    fig1, axes1 = plt.subplots(
        n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True
    )
    axes1 = np.array(axes1).reshape(-1)

    for ch_idx, ch_name in enumerate(ch_names):
        ax = axes1[ch_idx]
        has_any = False
        for band_name in band_names:
            col = f"{ch_name}_{band_name}"
            if col not in df_bands.columns:
                continue
            y = df_bands[col].values
            if np.all(np.isnan(y)):
                continue
            ax.plot(time, np.log10(y), label=band_name.upper(), linewidth=1.5, alpha=0.9)
            has_any = True
        ax.set_title(ch_name)
        ax.grid(True, alpha=0.3)
        if has_any:
            ax.legend(fontsize=8)

    # удалить пустые оси
    for idx in range(n_channels, len(axes1)):
        fig1.delaxes(axes1[idx])

    axes1[-1].set_xlabel("Время, с")
    fig1.suptitle("Ритмы по каналам", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- 2. каждый ритм по всем каналам (4 subplot) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()

    for i, band_name in enumerate(band_names[:4]):  # максимум 4 ритма
        ax = axes2[i]
        has_any = False
        for ch_name in ch_names:
            col = f"{ch_name}_{band_name}"
            if col not in df_bands.columns:
                continue
            y = df_bands[col].values
            if np.all(np.isnan(y)):
                continue
            ax.plot(time, np.log10(y), label=ch_name, alpha=0.7)
            has_any = True
        ax.set_title(f"{band_name.upper()} по каналам")
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Мощность")
        ax.grid(True, alpha=0.3)
        if has_any:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if dir_name is not None and resp_name is not None:
        fig1.savefig(
            f"{dir_name}/{resp_name}_channel_powers_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            f"{dir_name}/{resp_name}_rythms_powers_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    

# def compute_band_powers(
#     raw,
#     bands=None,
#     min_clean_ratio=0.5,
#     bad_prefix="bad",
# ):
#     """
#     Рассчитывает мощности ритмов за весь интервал записи.

#     Возвращает DataFrame:
#         строки  - каналы (только хорошие eeg‑каналы),
#         столбцы - ритмы (ключи из `bands`).
#     """
#     if bands is None:
#         bands = {
#             "theta": (4, 8),
#             "alpha": (8, 13),
#             "beta":  (13, 30),
#             "smr":   (12, 15),
#         }

#     sfreq = raw.info["sfreq"]
#     picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#     ch_names = np.array(raw.ch_names)[picks]

#     # Маска артефактов так же, как в sliding_band_powers_to_df_per_channel
#     bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)

#     # DataFrame: строки - каналы, столбцы - полосы
#     df = pd.DataFrame(index=ch_names, columns=bands.keys(), dtype=float)

#     # Берём весь интервал по времени
#     data_all, _ = raw[picks, :]

#     for ci, ch_name in enumerate(ch_names):
#         window_bad = bad_masks[ci, :]          # (n_times,)
#         clean_samples = ~window_bad
#         clean_ratio = clean_samples.sum() / len(clean_samples)
#         if clean_ratio < min_clean_ratio:
#             # Слишком много артефактов — оставляем NaN
#             continue

#         x = data_all[ci, clean_samples]
#         # как и в скользящем варианте: минимум 0.5 секунды чистых данных
#         if x.size < sfreq * 0.5:
#             continue

#         for band_name, (fmin, fmax) in bands.items():
#             val = compute_band_power_array(x, sfreq, fmin, fmax)
#             # нормировка на ширину полосы, как у вас в sliding_...
#             val = val / (fmax - fmin)
#             df.at[ch_name, band_name] = val

#     return df


# def compute_fooof_metrics(
#     raw,
#     freq_range=(1, 40),
#     peak_width_limits=(1.0, 6.0),
#     max_n_peaks=6,
#     min_clean_ratio=0.5,
#     bad_prefix="bad",
# ):
#     """
#     FOOOF‑метрики за весь интервал.
#     Строки - каналы, столбцы:
#       n_peaks,
#       aperiodic_offset, aperiodic_exponent, aperiodic_knee,
#       peak1_f/amp/width, ..., peakN_f/amp/width.
#     """
#     sfreq = raw.info["sfreq"]
#     picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#     ch_names = np.array(raw.ch_names)[picks]

#     bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)
#     data_all, _ = raw[picks, :]

#     clean_data = []
#     clean_ch_names = []
#     for ci, ch_name in enumerate(ch_names):
#         window_bad = bad_masks[ci, :]
#         clean_samples = ~window_bad
#         clean_ratio = clean_samples.sum() / len(clean_samples)
#         if clean_ratio < min_clean_ratio:
#             continue
#         x = data_all[ci, clean_samples]
#         if x.size < sfreq * 0.5:
#             continue
#         clean_data.append(x)
#         clean_ch_names.append(ch_name)

#     if len(clean_data) == 0:
#         cols = ["n_peaks", "aperiodic_offset", "aperiodic_exponent", "aperiodic_knee"]
#         for k in range(1, max_n_peaks + 1):
#             cols += [f"peak{k}_f", f"peak{k}_amp", f"peak{k}_width"]
#         return pd.DataFrame(columns=cols)

#     # выравниваем длину
#     min_len = min(len(x) for x in clean_data)
#     clean_data = np.stack([x[:min_len] for x in clean_data], axis=0)

#     # PSD
#     psds = []
#     for row in clean_data:
#         psd_row, freqs = psd_array_welch(
#             row[np.newaxis, :],
#             sfreq=sfreq,
#             fmin=freq_range[0],
#             fmax=freq_range[1],
#             n_per_seg=min(int(sfreq * 2), len(row)),
#             average="mean",
#             verbose=False,
#         )
#         psds.append(psd_row[0])
#     psds = np.stack(psds, axis=0)

#     # FOOOF
#     fg = FOOOFGroup(
#         peak_width_limits=peak_width_limits,
#         max_n_peaks=max_n_peaks,
#         verbose=False,
#     )
#     fg.fit(freqs, psds, freq_range=freq_range)

#     # столбцы
#     cols = ["n_peaks", "aperiodic_offset", "aperiodic_exponent", "aperiodic_knee"]
#     for k in range(1, max_n_peaks + 1):
#         cols += [f"peak{k}_f", f"peak{k}_amp", f"peak{k}_width"]

#     features = pd.DataFrame(index=clean_ch_names, columns=cols, dtype=float)

#     # aperiodic
#     ap = fg.get_params("aperiodic_params")
#     features["aperiodic_offset"] = ap[:, 0]
#     features["aperiodic_exponent"] = ap[:, 1]
#     if ap.shape[1] > 2:
#         features["aperiodic_knee"] = ap[:, 2]

#     # # peaks: list of arrays по каналам
#     # peak_params_list = fg.get_params("peak_params")
#     # print(peak_params_list)

#     # for ch_i, ch_name in enumerate(clean_ch_names):
#     #     peaks = peak_params_list[ch_i]  # (n_peaks_ch, 3) или (0,)
#     #     if peaks.size == 0:
#     #         features.at[ch_name, "n_peaks"] = 0
#     #         continue

#     #     if peaks.ndim == 1:
#     #         peaks = peaks[np.newaxis, :]

#     #     n_here = min(peaks.shape[0], max_n_peaks)
#     #     features.at[ch_name, "n_peaks"] = peaks.shape[0]

#     #     for k in range(n_here):
#     #         features.at[ch_name, f"peak{k+1}_f"] = peaks[k, 0]
#     #         features.at[ch_name, f"peak{k+1}_amp"] = peaks[k, 1]
#     #         if peaks.shape[1] > 2:
#     #             features.at[ch_name, f"peak{k+1}_width"] = peaks[k, 2]
    
#     peak_params_all = fg.get_params("peak_params")  # shape: (n_all_peaks, 4)
#     # столбцы: [CF, PW, BW, CH]
#     if peak_params_all.size == 0:
#         # ни одного пика
#         features["n_peaks"] = 0
#         return features
    
#     cf = peak_params_all[:, 0]
#     pw = peak_params_all[:, 1]
#     bw = peak_params_all[:, 2]
#     ch_idx = peak_params_all[:, 3].astype(int)
    
#     for ch_i, ch_name in enumerate(clean_ch_names):
#         # выбираем все пики этого канала
#         mask = ch_idx == ch_i
#         peaks_cf = cf[mask]
#         peaks_pw = pw[mask]
#         peaks_bw = bw[mask]
    
#         n_ch_peaks = peaks_cf.size
#         features.at[ch_name, "n_peaks"] = n_ch_peaks
    
#         if n_ch_peaks == 0:
#             continue
    
#         n_here = min(n_ch_peaks, max_n_peaks)
#         for k in range(n_here):
#             features.at[ch_name, f"peak{k+1}_f"] = peaks_cf[k]
#             features.at[ch_name, f"peak{k+1}_amp"] = peaks_pw[k]
#             features.at[ch_name, f"peak{k+1}_width"] = peaks_bw[k]

#     return features


# def plot_fooof_fits_per_channel(
#     raw,
#     freq_range=(1, 40),
#     peak_width_limits=(1.0, 6.0),
#     max_n_peaks=6,
#     min_clean_ratio=0.5,
#     bad_prefix="bad",
# ):
#     sfreq = raw.info["sfreq"]
#     picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#     ch_names = np.array(raw.ch_names)[picks]

#     bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)
#     data_all, _ = raw[picks, :]

#     chan_data = []
#     chan_labels = []
#     for ci, ch_name in enumerate(ch_names):
#         window_bad = bad_masks[ci, :]
#         clean_samples = ~window_bad
#         clean_ratio = clean_samples.sum() / len(clean_samples)
#         if clean_ratio < min_clean_ratio:
#             continue
#         x = data_all[ci, clean_samples]
#         if x.size < sfreq * 0.5:
#             continue
#         chan_data.append(x)
#         chan_labels.append(ch_name)

#     n_ch = len(chan_data)
#     if n_ch == 0:
#         print("Нет каналов с достаточным количеством чистых данных.")
#         return

#     n_cols = 2
#     n_rows = int(np.ceil(n_ch / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows))
#     axes = np.array(axes).reshape(-1)

#     from fooof import FOOOF

#     for idx, (x, ch_name) in enumerate(zip(chan_data, chan_labels)):
#         ax = axes[idx]

#         psd_row, freqs = psd_array_welch(
#             x[np.newaxis, :],
#             sfreq=sfreq,
#             fmin=freq_range[0],
#             fmax=freq_range[1],
#             n_per_seg=min(int(sfreq * 2), len(x)),
#             average="mean",
#             verbose=False,
#         )
#         psd_row = psd_row[0]          # линейный PSD

#         fm = FOOOF(
#             peak_width_limits=peak_width_limits,
#             max_n_peaks=max_n_peaks,
#             verbose=False,
#         )
#         fm.fit(freqs, psd_row, freq_range=freq_range)   # сюда — линейный PSD

#         # для отображения переводим ОТДЕЛЬНО в log10
#         log_psd = np.log10(psd_row)
#         log_fit = fm.fooofed_spectrum_      # уже в log10

#         ax.plot(freqs, log_psd, label="PSD (log10)", color="k", alpha=0.7)
#         ax.plot(freqs, log_fit, label="FOOOF fit", color="r", linewidth=2)

#         ax.set_ylabel("Power (log10 µV²/Hz)")

#     for ax in axes[n_ch:]:
#         fig.delaxes(ax)

#     fig.suptitle("FOOOF fits per channel", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# from fooof import FOOOF, FOOOFGroup
# from mne.time_frequency import psd_array_welch

def compute_fooof_models(
    raw,
    freq_range=(1, 40),
    peak_width_limits=(1.0, 6.0),
    max_n_peaks=6,
    min_clean_ratio=0.5,
    bad_prefix="bad",
):
    """
    Считает PSD и FOOOF-модели для всех каналов.
    Возвращает:
      freqs            : 1D массив частот
      psds             : (n_chan, n_freqs) PSD (linear)
      fooof_group      : FOOOFGroup, обученный на всех каналах
      clean_ch_names   : список имён каналов, по которым фит выполнен
    """
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = np.array(raw.ch_names)[picks]

    bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)
    data_all, _ = raw[picks, :]

    clean_data = []
    clean_ch_names = []
    for ci, ch_name in enumerate(ch_names):
        window_bad = bad_masks[ci, :]
        clean_samples = ~window_bad
        clean_ratio = clean_samples.sum() / len(clean_samples)
        if clean_ratio < min_clean_ratio:
            continue
        x = data_all[ci, clean_samples]
        if x.size < sfreq * 0.5:
            continue
        clean_data.append(x)
        clean_ch_names.append(ch_name)

    if len(clean_data) == 0:
        raise RuntimeError("Нет каналов с достаточным количеством чистых данных для FOOOF")

    # выравниваем длину
    min_len = min(len(x) for x in clean_data)
    clean_data = np.stack([x[:min_len] for x in clean_data], axis=0)  # (n_ch, n_times)

    # PSD для всех каналов
    psds = []
    for row in clean_data:
        psd_row, freqs = psd_array_welch(
            row[np.newaxis, :],
            sfreq=sfreq,
            fmin=freq_range[0],
            fmax=freq_range[1],
            n_per_seg=min(int(sfreq * 2), len(row)),
            average="mean",
            verbose=False,
        )
        psds.append(psd_row[0])
    psds = np.stack(psds, axis=0)  # (n_ch, n_freqs)

    # FOOOFGroup (fit один раз)
    fg = FOOOFGroup(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        verbose=False,
    )
    fg.fit(freqs, psds, freq_range=freq_range)

    return freqs, psds, fg, clean_ch_names


def fooof_and_bandpowers_to_df(
    freqs,
    psds,
    fg,
    clean_ch_names,
    bands=None,
    max_n_peaks=6,
):
    """
    Строит общий DataFrame:
      строки  - каналы,
      столбцы - FOOOF (n_peaks, aperiodic_*) + мощности по полосам (theta/alpha/...).

    bands: dict, например
        {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "smr": (12, 15)}
    """
    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
            "smr":   (12, 15),
        }

    # базовые колонки
    cols = ["n_peaks", "aperiodic_offset", "aperiodic_exponent", "aperiodic_knee"]
    for k in range(1, max_n_peaks + 1):
        cols += [f"peak{k}_f", f"peak{k}_amp", f"peak{k}_width"]
    # колонки для полос
    cols += list(bands.keys())

    features = pd.DataFrame(index=clean_ch_names, columns=cols, dtype=float)

    # ----- FOOOF aperiodic -----
    ap = fg.get_params("aperiodic_params")
    features["aperiodic_offset"] = ap[:, 0]
    features["aperiodic_exponent"] = ap[:, 1]
    if ap.shape[1] > 2:
        features["aperiodic_knee"] = ap[:, 2]

    # ----- FOOOF peaks (общий массив CF, PW, BW, CH) -----
    peak_params_all = fg.get_params("peak_params")  # (n_all_peaks, 4)
    if peak_params_all.size == 0:
        features["n_peaks"] = 0
    else:
        cf = peak_params_all[:, 0]
        pw = peak_params_all[:, 1]
        bw = peak_params_all[:, 2]
        ch_idx = peak_params_all[:, 3].astype(int)

        for ch_i, ch_name in enumerate(clean_ch_names):
            mask = ch_idx == ch_i
            peaks_cf = cf[mask]
            peaks_pw = pw[mask]
            peaks_bw = bw[mask]

            n_ch_peaks = peaks_cf.size
            features.at[ch_name, "n_peaks"] = n_ch_peaks

            if n_ch_peaks == 0:
                continue

            n_here = min(n_ch_peaks, max_n_peaks)
            for k in range(n_here):
                features.at[ch_name, f"peak{k+1}_f"] = peaks_cf[k]
                features.at[ch_name, f"peak{k+1}_amp"] = peaks_pw[k]
                features.at[ch_name, f"peak{k+1}_width"] = peaks_bw[k]

    # ----- мощности по полосам из PSD -----
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            continue
        # интеграл PSD по частоте → мощность в полосе
        band_power = np.trapz(psds[:, mask], freqs[mask], axis=1)
        features.loc[clean_ch_names, band_name] = band_power / (fmax-fmin)

    return features


def plot_fooof_fits_with_bands(freqs, psds, fg, clean_ch_names, df_fooof, bands=None):
    """
    freqs, psds, fg, clean_ch_names = compute_fooof_models(...)
    df_fooof = fooof_and_bandpowers_to_df(...)

    df_fooof: index=каналы, columns=ритмы (theta/alpha/...),
              значения — средние мощности (линейные).
    """
    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
            "smr":   (12, 15),
        }

    band_names = list(bands.keys())
    log_psds = np.log10(psds)

    n_ch = len(clean_ch_names)
    n_cols = 2
    n_rows = int(np.ceil(n_ch / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, ch_name in enumerate(clean_ch_names):
        ax = axes[idx]

        # 1) PSD
        ax.plot(freqs, log_psds[idx],
                color="k", alpha=0.7, label="PSD (log10)")

        # 2) FOOOF‑fit из группы
        fm = fg.get_fooof(idx, regenerate=True)      # восстановить модель[web:215]
        fooof_fit = fm.fooofed_spectrum_             # log10‑модель спектра[web:207]
        ax.plot(freqs, fooof_fit,
                color="r", linewidth=2, label="FOOOF fit")

        # 3) кусочно‑постоянная кривая средних мощностей из df_fooof
        x_band, y_band = [], []
        for band_name in band_names:
            if band_name not in df_fooof.columns:
                continue
            if ch_name not in df_fooof.index:
                continue
            val_lin = df_fooof.loc[ch_name, band_name]   # линейная мощность[file:226]
            y = np.log10(val_lin + 1e-20)

            fmin, fmax = bands[band_name]
            x_band.extend([fmin, fmax])
            y_band.extend([y, y])

        if x_band:
            ax.plot(x_band, y_band,
                    color="b", linewidth=2, alpha=0.4,
                    label="Band power (log10)")

        # масштаб по PSD
        y_min = log_psds[idx].min() - 0.5
        y_max = log_psds[idx].max() + 0.5
        ax.set_ylim(y_min, y_max)

        ax.set_title(ch_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (log10)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n_ch:]:
        fig.delaxes(ax)

    fig.suptitle("FOOOF fits & band powers per channel", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig.savefig(
        "fooof_plot.png",
        dpi=300,
        bbox_inches="tight",
    )



