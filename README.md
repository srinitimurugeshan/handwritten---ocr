# AI Handwritten Text Recognition – Tamil & English

Static website (HTML + CSS + JS, Tesseract.js). No server needed.

## Run locally
Open a terminal in this folder: `python3 -m http.server 8000` → open http://localhost:8000

## Deploy on GitHub Pages
1. Create a new GitHub repo and upload `index.html`, `style.css`, `app.js`, `README.md`.
2. Repo **Settings → Pages → Source: Deploy from a branch → main / (root) → Save**.
3. After ~1 minute the site is live at `https://<username>.github.io/<repo>/`.

## Pipeline (matches the methodology diagram)
Upload → Quality analysis → Adaptive preprocessing & noise removal → Handwritten text detection →
Line/word segmentation → Language identification → Tamil+English recognition → Confidence & uncertain words →
Spell correction → Editable text + original/processed comparison → Translation (MyMemory API) → Export TXT/PDF/Word
