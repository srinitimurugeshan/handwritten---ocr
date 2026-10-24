# Handwritten Text Recognition and Multilingual Translation System

A web application that reads handwritten **Tamil and English** text from an image, lets you correct it, translates it into another language and exports the result.

**Live demo:** https://srinitimurugeshan.github.io/handwritten---ocr/

Mini project by **Sriniti M** (24UVA055), III B.Voc Artificial Intelligence and Machine Learning.

## Features

- Upload a handwritten image (JPG or PNG) in Tamil, English or both
- Image quality analysis (brightness, contrast, sharpness) with warnings
- Automatic cleaning: shadow flattening, noise removal, adaptive thresholding, ruled-line removal
- Line and word detection, shown as boxes on the cleaned image
- Automatic language identification (Tamil or English)
- Word-level confidence; uncertain words are highlighted for review
- Editable output with an original vs cleaned image comparison slider
- English spell correction
- Full-page translation into Tamil, English, Hindi, Telugu, Malayalam, Kannada, French, Spanish or German
- Export as TXT, Word or PDF
- Built-in accuracy check: type the correct text and get word and character accuracy, with a table to copy into a report
- Automatic fallback to Tesseract if the Gemini service is unavailable
- Camera button on phones

## How it works

1. Image upload
2. Image quality analysis
3. Adaptive preprocessing and noise removal
4. Handwritten text detection
5. Line and word segmentation
6. Language identification
7. Tamil and English recognition
8. Confidence score and uncertain-word detection
9. Spell correction
10. Editable text with original and cleaned image comparison
11. Translation
12. Export as TXT, PDF or Word

## Technologies

- HTML, CSS and JavaScript (runs fully in the browser, hosted on GitHub Pages)
- [Tesseract.js](https://github.com/naptha/tesseract.js) for OCR (Tamil and English language data)
- Canvas API for image preprocessing
- LanguageTool API for English spell correction
- MyMemory API for translation
- Optional: Google Gemini API for more accurate recognition of cursive and Tamil handwriting and for translation

## How to use

1. Open the live demo and click **Choose a handwritten image**.
2. (Optional) Open **Gemini AI engine**, paste your own free API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Click **Run recognition**.
4. Edit the recognised text, choose a target language and translate.
5. Download the result as TXT, Word or PDF (for PDF choose **Save as PDF** in the print dialog).

The Gemini key is stored only in your browser and is never part of this repository. With a key, the image is sent to Google for processing.

## Run locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000

## Limitations

- Tesseract is designed mainly for printed text, so accuracy on handwriting, especially cursive and Tamil handwriting, is limited without the Gemini option.
- Spell correction is available for English only.
- The free translation service has a daily usage limit.
- Clear, dark writing on plain paper gives the best results.

## Future work

- Measure accuracy on a larger set of handwriting samples
- Better recognition of cursive handwriting
- Tamil spell correction
- Multi-page upload and automatic deskew
