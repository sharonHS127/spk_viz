## Channel Order (`--order`)

The channel layout is automatically determined based on the number of channels:

- `1` = Mono
- `2` = Stereo
- `6` = 5.1 Surround
- `8` = 7.1 Surround

If the channel order in your audio file differs from the default layout, use `--order` to specify the correct channel order.

### Supported Channel Names

| Name | Description | Azimuth |
|------|-------------|---------|
| FL | Front Left | -30° |
| FR | Front Right | +30° |
| FC | Front Center | 0° |
| LFE | Low-Frequency Effects (Subwoofer) | N/A |
| SL | Side Left | 7.1: -90° / 5.1: -110° |
| SR | Side Right | 7.1: +90° / 5.1: +110° |
| BL | Back Left | -150° (7.1 only) |
| BR | Back Right | +150° (7.1 only) |

### Default Channel Order and Presets

| Channels | `wav` (Default) | `film` |
|----------|----------------|--------|
| 1 | FC | FC |
| 2 | FL, FR | FL, FR |
| 6 (5.1) | FL, FR, FC, LFE, SL, SR | Same as `wav` |
| 8 (7.1) | FL, FR, FC, LFE, BL, BR, SL, SR | FL, FR, FC, LFE, SL, SR, BL, BR |

### Examples

```bash
# 7.1 surround — default order (WAV / FFmpeg convention)
python spk_viz.py my_71.wav

# 7.1 surround — cinema / Dolby order
python spk_viz.py my_71.wav --order film

# Custom channel order (comma-separated, case-insensitive)
python spk_viz.py my_71.wav --order FL,FR,FC,LFE,SL,SR,BL,BR

# Custom 5.1 order (e.g., FC as the first channel)
python spk_viz.py my_51.wav --order FC,FL,FR,SL,SR,LFE

# Stereo with left and right channels swapped
python spk_viz.py my_stereo.wav --order FR,FL
```

### Notes

- The number of channel names in a custom layout must match the number of channels in the audio file.
- Only channel names supported by the selected layout are allowed. For example, `BL` and `BR` are not available in 5.1.
- An incorrect channel order will cause the wrong speaker to light up in the visualization. Use `--make-test` to generate a test signal and verify the channel mapping before processing your audio files.
