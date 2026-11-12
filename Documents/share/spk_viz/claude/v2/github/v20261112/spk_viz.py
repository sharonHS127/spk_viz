#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spk_viz.py — Visualization and headphone preview for mono, stereo, 5.1, and 7.1 WAV files.

Automatically detects the channel layout based on the number of channels:
1 = Mono (FC), 2 = Stereo (FL, FR ±30°), 6 = 5.1, 8 = 7.1.

Input: A WAV file.
Outputs:
  <name>_binaural.wav   Binaural headphone preview, treating all 8 channels
                        as speakers positioned in a standard 7.1 layout.

  <name>_viz.mp4        Top-down animated visualization featuring:
                        - Speaker illumination based on audio levels
                        - Directional energy indicator
                        - Per-channel level meters
                        - Timeline
                        The MP4 includes the binaural preview audio.

Requirements:
  numpy, scipy, matplotlib, ffmpeg (must be available in PATH).
  Optional: h5py (for reading SOFA-format HRIR files).

Usage:
  python spk_viz.py my_71.wav

  python spk_viz.py my_71.wav --order film
      # Channel order: L R C LFE Ls Rs Lb Rb

  python spk_viz.py my_71.wav --start 30 --duration 20
      # Process a specific segment only.

  python spk_viz.py my_71.wav --sofa HRIR_L2702.sofa
      # Use measured HRIR data from a KU100 dummy head.

  python spk_viz.py --make-test test_71.wav
      # Generate a rotating surround sound test signal.

Note:
  The binaural preview simulates an ideal 7.1 speaker setup. It does not
  account for the up-firing speakers built into the soundbar, the actual
  placement of the rear surround speakers, or room reflections.

  It is intended for checking channel mapping and spatial perception,
  not for predicting the sound signature of a real home theater system.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from math import gcd

import numpy as np
from scipy import signal
from scipy.io import wavfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- Speaker layout
# Azimuth: 0 = front, positive = right, negative = left (degrees);
# None = LFE (non-directional).
AZ_71 = {"FL": -30, "FR": 30, "FC": 0, "LFE": None,
         "SL": -90, "SR": 90, "BL": -150, "BR": 150}
AZ_51 = {"FL": -30, "FR": 30, "FC": 0, "LFE": None, "SL": -110, "SR": 110}

# Channel order in the file
ORDERS = {
    "wav": "FL,FR,FC,LFE,BL,BR,SL,SR",    # Standard 7.1 channel order for WAV / ffmpeg (default)
    "film": "FL,FR,FC,LFE,SL,SR,BL,BR",   # cinema / Dolby：L R C LFE Ls Rs Lrs Rrs
}


AZ_20 = {"FL": -30, "FR": 30}
AZ_10 = {"FC": 0}
LAYOUT_NAMES = {1: "1.0 mono", 2: "2.0 stereo", 6: "5.1", 8: "7.1"}


def get_layout(order, nch):
    table = {8: AZ_71, 6: AZ_51, 2: AZ_20, 1: AZ_10}.get(nch)
    if table is None:
        sys.exit(f"Supported channel counts are 1 (mono), 2 (stereo), 6 (5.1), and 8 (7.1). This file has {nch} channels.")
    if order in ("wav", "film"):
        if nch == 8:
            s = ORDERS[order]
        elif nch == 6:
            s = "FL,FR,FC,LFE,SL,SR"
        elif nch == 2:
            s = "FL,FR"
        else:
            s = "FC"
    else:
        s = order
    names = [n.strip().upper() for n in s.split(",")]
    if len(names) != nch or any(n not in table for n in names):
        sys.exit(fㄇ)
    return [(n, table[n]) for n in names]


# ---------------------------------------------------------------- Read and write audio files
def read_wav(path):
    try:
        fs, x = wavfile.read(path)
    except Exception:
        tmp = os.path.join(tempfile.mkdtemp(), "conv.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-c:a", "pcm_f32le", tmp], check=True)
        fs, x = wavfile.read(tmp)
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return fs, x


# ---------------------------------------------------------------- Binaural rendering
def head_model_irs(az_deg, fs, n=512, a=0.0875, c=343.0):
    """Fallback model when HRIR is unavailable: Woodworth ITD + Brown-Duda head shadowing + rear high-frequency attenuation. Returns: (left_ir, right_ir)."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    w = 2 * np.pi * f
    w0 = c / a
    az = np.radians(az_deg)
    lat = np.arcsin(np.sin(az))  # Front-back folded azimuth angle
    itd = a / c * (abs(lat) + np.sin(abs(lat)))
    rear = np.clip((abs(az_deg) - 60) / 120.0, 0, 1)
    shelf = 1.0 - (1.0 - 10 ** (-4.0 / 20)) * rear / (1.0 + (3000.0 / np.maximum(f, 1.0)) ** 4)
    irs = []
    for sign in (-1.0, +1.0):  # left and right ear
        side = np.sin(az) * sign  # >0：the sound source is on this ear's side
        theta = np.degrees(np.arccos(np.clip(side, -1, 1)))  # Angle relative to the interaural axis
        alpha = 1.05 + 0.95 * np.cos(np.radians(theta * 180.0 / 150.0))
        H = (1 + 1j * alpha * w / (2 * w0)) / (1 + 1j * w / (2 * w0)) * shelf
        tau = 0.0004 + (itd if side < 0 else 0.0)  # itd
        irs.append(np.fft.irfft(H * np.exp(-1j * w * tau), n))
    return irs[0], irs[1]


def load_sofa(path, fs):
    import h5py  # Lazy import; missing dependency doesn't affect the fallback model.
    with h5py.File(path, "r") as h:
        ir = np.array(h["Data.IR"])
        sfs = float(np.array(h["Data.SamplingRate"]).ravel()[0])
        pos = np.array(h["SourcePosition"])
        typ = h["SourcePosition"].attrs.get("Type", "spherical")
    if isinstance(typ, (bytes, np.bytes_)):
        typ = typ.decode()
    typ = str(typ).lower()
    if typ.startswith("cart"):
        xyz = pos[:, :3]
        unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    else:  # SOFA spherical coordinates: counter-clockwise azimuth is positive (0 = front, 90 = left).
        az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
        unit = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    if int(round(sfs)) != int(fs):
        g = gcd(int(fs), int(round(sfs)))
        ir = signal.resample_poly(ir, int(fs) // g, int(round(sfs)) // g, axis=2)
    return unit, ir


def sofa_irs(az_deg, sofa):
    unit, ir = sofa
    a = np.radians(-az_deg)  # right-positive
    d = np.array([np.cos(a), np.sin(a), 0.0])
    k = int(np.argmax(unit @ d))
    return ir[k, 0, :], ir[k, 1, :]


def render_binaural(x, fs, spk, sofa=None, lfe_gain=1.0):
    n = x.shape[0]
    out = np.zeros((n, 2))
    for i, (name, az) in enumerate(spk):
        if az is None:  # LFE
            out += lfe_gain * 0.7071 * x[:, [i]]
            continue
        irl, irr = sofa_irs(az, sofa) if sofa is not None else head_model_irs(az, fs)
        out[:, 0] += signal.fftconvolve(x[:, i], irl)[:n]
        out[:, 1] += signal.fftconvolve(x[:, i], irr)[:n]
    return out


def write_pcm16(path, fs, y):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.98:
        y = y * (0.98 / peak)
        print(f"  （Binaural preview peak {peak:.2f} exceeded full scale, reduced overall by {20*np.log10(peak/0.98):.1f} dB）")
    wavfile.write(path, int(fs), (np.clip(y, -1, 1) * 32767).astype(np.int16))


# ---------------------------------------------------------------- analysis
def analyze(x, fs, spk, fps, win_ms):
    n, nch = x.shape
    nfr = int(n / fs * fps)
    cs = np.vstack([np.zeros((1, nch)), np.cumsum(x ** 2, axis=0)])
    centers = (np.arange(nfr) + 0.5) * fs / fps
    half = int(fs * win_ms / 2000.0)
    s = np.clip((centers - half).astype(int), 0, n)
    e = np.clip((centers + half).astype(int), 0, n)
    ms = (cs[e] - cs[s]) / np.maximum(e - s, 1)[:, None]
    db = 10 * np.log10(ms + 1e-12)  # RMS per channel（dBFS）

    di = [i for i, (_, az) in enumerate(spk) if az is not None]
    u = np.array([[np.sin(np.radians(spk[i][1])), np.cos(np.radians(spk[i][1]))] for i in di])
    E = ms[:, di]
    Esum = E.sum(1)
    v = (E @ u) / np.maximum(Esum, 1e-12)[:, None]  # Energy vector, length 0 (diffuse) to 1 (point source)
    total_db = 10 * np.log10(Esum + 1e-12)
    active = total_db > -60
    v[~active] = 0.0
    az_est = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
    conc = np.hypot(v[:, 0], v[:, 1])
    return dict(nfr=nfr, db=db, v=v, total_db=total_db, active=active,
                az=az_est, conc=conc, t=(np.arange(nfr) + 0.5) / fps)


def smooth_display(A, fps, release_db_s=24.0, k=0.55):
    db = A["db"].copy()
    vs = A["v"].copy()
    for i in range(1, A["nfr"]):
        db[i] = np.maximum(db[i], db[i - 1] - release_db_s / fps)
        vs[i] = k * vs[i - 1] + (1 - k) * vs[i]
    A["db_disp"] = db
    A["v_disp"] = vs


# ---------------------------------------------------------------- video
BG, FG, MUTED, ACCENT = "#0e1117", "#e5e7eb", "#6b7280", "#7dd3fc"


def render_video(out_mp4, spk, A, fps, W, H, audio_wav, title):
    nch = len(spk)
    cmap = plt.get_cmap("plasma")
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100, facecolor=BG)

    # --- top-down view
    ax = fig.add_axes([0.02, 0.27, 0.56, 0.68], facecolor=BG)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#2a303c", lw=1.2, ls=(0, (4, 4)), zorder=0)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color="#1c212b", lw=1.0, zorder=0)
    ax.plot(0.12 * np.cos(th), 0.12 * np.sin(th), color=FG, lw=1.5, zorder=3)  # head
    ax.fill([-0.04, 0.04, 0.0], [0.115, 0.115, 0.19], color=FG, zorder=3)      # nose（front = upward）

    pos = []
    for i, (name, az) in enumerate(spk):
        if az is None:
            pos.append((1.42, -1.20))
        else:
            pos.append((np.sin(np.radians(az)), np.cos(np.radians(az))))
    pos = np.array(pos)
    for i, (name, az) in enumerate(spk):
        if az is None:
            lx, ly, txt = pos[i][0], pos[i][1] - 0.50, f"{name} (sub)  ch{i + 1}"
        else:
            lx, ly = pos[i] * 1.46
            txt = f"{name}\nch{i + 1}"
        ax.text(lx, ly, txt, ha="center", va="center", fontsize=9, color=FG, linespacing=1.15)

    halo = ax.scatter(pos[:, 0], pos[:, 1], s=np.full(nch, 200.0), c=np.zeros(nch),
                      cmap=cmap, vmin=0, vmax=1, alpha=0.22, linewidths=0, zorder=1, clip_on=False)
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=np.full(nch, 200.0), c=np.zeros(nch),
                    cmap=cmap, vmin=0, vmax=1, edgecolors=MUTED, linewidths=1.0, zorder=2, clip_on=False)
    (shaft,) = ax.plot([0, 0], [0, 0], color=ACCENT, lw=4, solid_capstyle="round", zorder=4)
    (tip,) = ax.plot([0], [0], "o", color=ACCENT, ms=13, zorder=5)
    readout = ax.text(0, -1.68, "", ha="center", va="center", fontsize=11, color=ACCENT)

    # --- level meter
    axb = fig.add_axes([0.68, 0.33, 0.28, 0.57], facecolor=BG)
    ypos = np.arange(nch)[::-1]
    bars = axb.barh(ypos, np.zeros(nch), height=0.62, color=cmap(0.0))
    axb.set_xlim(0, 1.22)
    axb.set_ylim(-0.6, nch - 0.4)
    axb.set_yticks(ypos)
    axb.set_yticklabels([f"{n}  ch{i + 1}" for i, (n, _) in enumerate(spk)], fontsize=10, color=FG)
    axb.set_xticks([0, 1 / 3, 2 / 3, 1])
    axb.set_xticklabels(["-60", "-40", "-20", "0 dBFS"], fontsize=8)
    axb.grid(axis="x", color="#1f2530", lw=0.8)
    axb.set_axisbelow(True)
    for sp in axb.spines.values():
        sp.set_visible(False)
    axb.tick_params(length=0)
    db_txt = [axb.text(0, ypos[i], "", va="center", fontsize=9, color=FG) for i in range(nch)]
    fig.text(0.68, 0.915, "Channel level (RMS)", fontsize=10, color=MUTED)
    fig.text(0.68, 0.275, "Arrow = energy direction (front = up)\n"
             "Focus 1 = single point, 0 = diffuse", fontsize=8.5, color=MUTED, va="top")
    fig.text(0.02, 0.965, title, fontsize=12, color=FG, va="center")
    time_txt = fig.text(0.58, 0.965, "", fontsize=12, color=ACCENT, va="center", ha="right")

    # --- Timeline (Master Volume, Azimuth)
    T = A["nfr"] / fps
    t = A["t"]
    ax1 = fig.add_axes([0.08, 0.175, 0.88, 0.065], facecolor=BG)
    ax1.fill_between(t, -60, np.clip(A["total_db"], -60, 0), color="#374151", lw=0)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-60, 0)
    ax1.set_yticks([-60, -30, 0])
    ax1.tick_params(labelsize=8, length=0, labelbottom=False)
    ax1.set_ylabel("dBFS", fontsize=8, color=MUTED)
    ax2 = fig.add_axes([0.08, 0.04, 0.88, 0.115], facecolor=BG)
    m = A["active"]
    lvl = np.clip((A["total_db"][m] + 60) / 60.0, 0, 1)
    ax2.scatter(t[m], A["az"][m], s=5, c=lvl, cmap=cmap, vmin=0, vmax=1, linewidths=0)
    ax2.set_xlim(0, T)
    ax2.set_ylim(-190, 190)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_yticklabels(["rear", "L", "front", "R", "rear"])
    ax2.tick_params(labelsize=8, length=0)
    ax2.set_xlabel("time (s)", fontsize=8, color=MUTED, labelpad=1)
    for a_ in (ax1, ax2):
        for sp in a_.spines.values():
            sp.set_color("#2a303c")
        a_.grid(color="#1a1f29", lw=0.6)
    cur1 = ax1.axvline(0, color=ACCENT, lw=1.2)
    cur2 = ax2.axvline(0, color=ACCENT, lw=1.2)

    # --- ffmpeg
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", audio_wav]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20"]
    if audio_wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    norm_all = np.clip((A["db_disp"] + 60.0) / 60.0, 0, 1)
    t0 = time.time()
    for i in range(A["nfr"]):
        nm = norm_all[i]
        sizes = 150 + 3200 * nm ** 1.2
        sc.set_sizes(sizes)
        sc.set_array(nm)
        halo.set_sizes(sizes * 2.4)
        halo.set_array(nm)
        for b, w_, txt, k in zip(bars, nm, db_txt, range(nch)):
            b.set_width(w_)
            b.set_facecolor(cmap(0.15 + 0.85 * w_))
            txt.set_x(w_ + 0.02)
            txt.set_text(f"{A['db_disp'][i, k]:.0f}" if A["db_disp"][i, k] > -60 else "")
        vx, vy = A["v_disp"][i]
        mag = float(np.hypot(vx, vy))
        if A["total_db"][i] > -60 and mag > 0.04:
            L = 0.95 * min(mag, 1.0)
            ux, uy = vx / mag, vy / mag
            shaft.set_data([0, ux * L], [0, uy * L])
            tip.set_data([ux * L], [uy * L])
            shaft.set_alpha(1.0)
            tip.set_alpha(1.0)
            readout.set_text(f"azimuth {np.degrees(np.arctan2(vx, vy)):+4.0f}°   focus {mag:.2f}")
        else:
            shaft.set_alpha(0.0)
            tip.set_alpha(0.0)
            readout.set_text("diffuse / silent" if A["total_db"][i] > -60 else "silent")
        tt = t[i]
        cur1.set_xdata([tt, tt])
        cur2.set_xdata([tt, tt])
        time_txt.set_text(f"{tt:6.2f} s")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break
        if (i + 1) % 150 == 0:
            print(f"  frame {i + 1}/{A['nfr']}  ({time.time() - t0:.0f}s)")
    try:
        proc.stdin.close()
    except Exception:
        pass
    rc = proc.wait()
    plt.close(fig)
    if rc != 0:
        sys.exit("ffmpeg failed. Please check the error message above.")


# ---------------------------------------------------------------- for testing
def make_test(path, order="wav", fs=48000, dur=14.0):
    """0–8s: noise pans clockwise 360°; 8–10s: SL only; 10–12s: BR only; 12–14s: diffuse; LFE pulses throughout."""
    spk = get_layout(order, 8)
    rng = np.random.default_rng(0)
    n = int(fs * dur)
    t = np.arange(n) / fs

    def pink(m):
        W_ = np.fft.rfft(rng.standard_normal(m))
        f = np.fft.rfftfreq(m, 1 / fs)
        f[0] = f[1]
        p = np.fft.irfft(W_ / np.sqrt(f), m)
        return p / np.max(np.abs(p))

    x = np.zeros((n, 8))
    di = [i for i, (_, az) in enumerate(spk) if az is not None]
    azs = np.array([spk[i][1] for i in di], float)
    o = np.argsort(azs)
    s, idxs = azs[o], np.array(di)[o]
    n1 = int(8 * fs)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t[:n1]))
    sig = pink(n1) * env * 0.5
    azp = ((360.0 * t[:n1] / 8.0 - s[0]) % 360.0) + s[0]
    bext = np.append(s, s[0] + 360.0)
    j = np.searchsorted(s, azp, side="right") - 1
    u = (azp - bext[j]) / (bext[j + 1] - bext[j])
    rows = np.arange(n1)
    x[rows, idxs[j]] += sig * np.cos(u * np.pi / 2)
    x[rows, idxs[(j + 1) % len(idxs)]] += sig * np.sin(u * np.pi / 2)

    names = [nm for nm, _ in spk]
    a, b = int(8 * fs), int(10 * fs)
    tone = 0.4 * np.sin(2 * np.pi * 1000 * t[a:b] + 3 * np.sin(2 * np.pi * 5 * t[a:b]))
    x[a:b, names.index("SL")] += tone
    a, b = int(10 * fs), int(12 * fs)
    x[a:b, names.index("BR")] += 0.5 * pink(b - a)
    a = int(12 * fs)
    for i in di:
        x[a:, i] += 0.15 * pink(n - a)
    x[:, names.index("LFE")] += 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * t)) * np.sin(2 * np.pi * 40 * t)
    wavfile.write(path, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print(f"Test audio generated: {path}（{dur:.0f} sec., 8-channel, order {order}）")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="7.1 wav Visualization + headphone binaural preview")
    ap.add_argument("wav", nargs="?", help="Input 7.1 (or 5.1) WAV file")
    ap.add_argument("--make-test", metavar="PATH", help="Generate rotating surround test audio and exit")
    ap.add_argument("-o", "--out-prefix", help="Output filename prefix (default: input filename)")
    ap.add_argument("-d", "--out-dir",
                    help="Output directory name or path; automatically created if it does not exist (default: <filename>_output next to the input file)")
    ap.add_argument("--order", default="wav",
                    help="order：wav（default: FL FR FC LFE BL BR SL SR）、film（FL FR FC LFE SL SR BL BR），"
                         "Or a comma-separated list (e.g., FL,FR,FC,LFE,SL,SR,BL,BR)")
    ap.add_argument("--sofa", help="SOFA format HRIR (e.g., HRIR_L2702.sofa for KU100); requires h5py")
    ap.add_argument("--lfe-gain", type=float, default=1.0, help="LFE gain for binaural preview (default: 1.0)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720", help="Video resolution, default: 1280x720")
    ap.add_argument("--win-ms", type=float, default=100.0, help="Level analysis window (ms)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (sec.)")
    ap.add_argument("--duration", type=float, default=None, help="Processing duration (sec)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-binaural", action="store_true")
    args = ap.parse_args()

    if args.make_test:
        make_test(args.make_test, args.order)
        return
    if not args.wav:
        ap.error("Please provide an input WAV file, or use --make-test to generate test audio.")

    fs, x = read_wav(args.wav)
    spk = get_layout(args.order, x.shape[1])
    a = int(args.start * fs)
    b = x.shape[0] if args.duration is None else min(x.shape[0], a + int(args.duration * fs))
    x = x[a:b]
    if x.shape[0] < fs // 10:
        sys.exit("Audio to process is too short.")
    print(f"input：{args.wav}  {x.shape[1]} channel  {fs} Hz  {x.shape[0] / fs:.1f} 秒")
    print("Channel mapping: " + "  ".join(f"ch{i + 1}={n}" for i, (n, _) in enumerate(spk)))

    base = args.out_prefix or os.path.splitext(os.path.basename(args.wav))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.wav)), base + "_output")
    os.makedirs(out_dir, exist_ok=True)  # Create directory if it does not exist
    print(f"Output directory: {out_dir}")
    stem = os.path.join(out_dir, base)
    bin_path = stem + "_binaural.wav"
    mp4_path = stem + "_viz.mp4"

    have_bin = False
    if not args.no_binaural:
        sofa = None
        if args.sofa:
            sofa = load_sofa(args.sofa, fs)
            print(f"Using SOFA HRIR: {args.sofa}")
        else:
            print("Using built-in simplified head model (add --sofa for higher realism)")
        y = render_binaural(x, fs, spk, sofa, args.lfe_gain)
        write_pcm16(bin_path, fs, y)
        have_bin = True
        print(f"Exported: {bin_path}")

    if not args.no_video:
        W, H = (int(v) for v in args.size.lower().split("x"))
        W, H = W - W % 2, H - H % 2
        A = analyze(x, fs, spk, args.fps, args.win_ms)
        smooth_display(A, args.fps)
        print(f"Rendering {A['nfr']} frames")
        render_video(mp4_path, spk, A, args.fps, W, H, bin_path if have_bin else None,
                     f"{os.path.basename(args.wav)}  —  {LAYOUT_NAMES[len(spk)]} speaker view")
        print(f"Exported: {mp4_path}")


if __name__ == "__main__":
    main()
