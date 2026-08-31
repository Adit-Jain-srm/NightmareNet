# Multilingual Keyboard Typo Distortion

Layout-aware typographical noise for English, German, French, Russian, Hindi,
and Arabic. Neighbors are taken from QWERTY / QWERTZ / AZERTY / Cyrillic /
Arabic / Devanagari grids (horizontal neighbors weighted higher than vertical).

## CLI

```bash
nightmarenet distort --type keyboard_typo --language german --text "Hallo Welt" --strength 0.5 --seed 42
nightmarenet distort --list-engines   # includes keyboard_typo
```

Optional `--keyboard-layout` overrides the language default (`qwertz` for German,
`azerty` for French, etc.).

## Config

```yaml
distortion:
  language: german
  keyboard_layout: qwertz   # optional override
```

## Custom layouts

```python
from nightmarenet.distortions.multilingual import register_layout, distort

register_layout("dvorak", ("pyfgcrl", "aoeuidhtns", "qjkxbmwvz"))
print(distort("hello", strength=0.5, seed=1, keyboard_layout="dvorak"))
```

Error types: replace (nearby key), insert (double press), delete, transpose.
`strength` maps to character error rate in `[0, 1]`.

### Known Limitations

- **Complex grapheme clusters:** Hindi/Arabic combining characters (vowel signs, virama) are split using Unicode `\X` grapheme segmentation, which handles most cases correctly. However, some edge cases with deeply nested combining sequences may produce unexpected results. Grapheme cluster preservation improvements are planned for v2.
