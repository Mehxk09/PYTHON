# dicionario.py — Dictionary Blueprint (separate from main app)
import os
from flask import Blueprint, render_template, send_file, jsonify, abort

dicionario_bp = Blueprint('dicionario', __name__)

LETTERS_DIR = os.path.join("dataset", "asl_alphabets")
WORDS_DIR = os.path.join("dataset", "psl_words")


def get_alphabet_data():
    """Get available letters and their first sample image."""
    letters = []
    if not os.path.exists(LETTERS_DIR):
        return letters

    for folder in sorted(os.listdir(LETTERS_DIR)):
        folder_path = os.path.join(LETTERS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        images = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if images:
            # Prefer images with "exemplo" in the name (user-chosen display image)
            exemplo = [f for f in images if "exemplo" in f.lower()]
            chosen = exemplo[0] if exemplo else images[0]
            letters.append({
                "letter": folder,
                "image": chosen,
                "count": len(images)
            })
    return letters


def get_words_data():
    """Get available words and their display video."""
    words = []
    if not os.path.exists(WORDS_DIR):
        return words

    for folder in sorted(os.listdir(WORDS_DIR)):
        folder_path = os.path.join(WORDS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        videos = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith('.mp4')])
        if videos:
            # Prefer videos with "exemplo" in the name (user-chosen display video)
            exemplo = [f for f in videos if "exemplo" in f.lower()]
            chosen = exemplo[0] if exemplo else videos[0]
            display_name = folder.replace("_", " ")
            if folder == "Nao":
                display_name = "Não"
            words.append({
                "word": folder,
                "display_name": display_name,
                "video": chosen,
                "count": len(videos)
            })
    return words


# ---- Routes ----

@dicionario_bp.route('/dicionario')
def dicionario():
    """Render the dictionary page."""
    letters = get_alphabet_data()
    words = get_words_data()
    return render_template('dicionario.html', letters=letters, words=words)


@dicionario_bp.route('/dicionario/image/<letter>/<filename>')
def serve_letter_image(letter, filename):
    """Serve a letter sample image from the dataset."""
    safe_letter = letter.upper()
    path = os.path.join(LETTERS_DIR, safe_letter, filename)
    if not os.path.exists(path):
        abort(404)
    mime = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
    return send_file(os.path.abspath(path), mimetype=mime)


@dicionario_bp.route('/dicionario/video/<word>/<filename>')
def serve_word_video(word, filename):
    """Serve a word sample video from the dataset."""
    path = os.path.join(WORDS_DIR, word, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(os.path.abspath(path), mimetype='video/mp4')






























