# AI Handwritten Text Recognition – Tamil & English

Static website (HTML + CSS + JS, Tesseract.js). No server needed.

🔗 **Live demo:** https://srinitimurugeshan.github.io/handwritten---ocr/

## Features
- Handwritten text recognition for Tamil and English
- Image preprocessing and noise removal
- Confidence score and uncertain word detection
- Spell correction
- Translation (MyMemory API)
- Export as TXT / PDF / Word

## Run locally
```bash
python3 -m http.server 8000
```
Then open http://localhost:8000

## Deploy on GitHub Pages
1. Upload `index.html`, `style.css`, `app.js`, `README.md`
2. Settings → Pages → Source: Deploy from a branch → main / (root) → Save
3. Site goes live at `https://<username>.github.io/<repo>/`

## Pipeline
Upload → Quality analysis → Adaptive preprocessing → Text detection → Line/word segmentation → Language identification → Tamil+English recognition → Confidence check → Spell correction → Editable text → Translation → Export
