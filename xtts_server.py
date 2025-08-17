#!/usr/bin/env python3
import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts = None

def ensure_model():
    global _tts
    if _tts is None:
        # Safe-load workaround for torch 2.6+
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
            try:
                add_safe_globals([XttsConfig])
            except Exception:
                pass
        except Exception:
            pass
        from TTS.api import TTS
        _tts = TTS(MODEL_ID)
    return _tts

@app.get('/health')
def health():
    try:
        ready = _tts is not None
        return jsonify({ 'status': 'ready' if ready else 'initializing', 'engine': 'xtts_v2' })
    except Exception:
        return jsonify({ 'status': 'initializing', 'engine': 'xtts_v2' })

@app.post('/api/tts')
def tts_api():
    data = request.get_json(silent=True) or {}
    text = data.get('text') or ''
    speaker = (data.get('speaker') or data.get('speaker_id') or 'kelly').lower()
    language = data.get('language') or data.get('language_id') or 'en'
    if not text:
        return jsonify({ 'error': 'No text provided' }), 400

    tts = ensure_model()

    refs = {
        'kelly': os.path.join(os.path.dirname(__file__), 'dist', 'reference_kelly.wav'),
        'ken': os.path.join(os.path.dirname(__file__), 'dist', 'reference_ken_mono16k.wav')
    }
    ref_wav = refs.get(speaker)
    if not (ref_wav and os.path.exists(ref_wav)):
        return jsonify({ 'error': f'reference wav not found for {speaker}' }), 500

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    tts.tts_to_file(text=text, speaker_wav=ref_wav, language=language, file_path=wav_path)

    # Mastering and mp3 encode
    out_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
    ff = (
        f"ffmpeg -y -hide_banner -loglevel error -i '{wav_path}' "
        "-filter:a highpass=f=80,acompressor=threshold=-20dB:ratio=2:attack=5:release=120,loudnorm=I=-23:LRA=7:TP=-1.0 "
        "-ar 48000 -ac 1 -c:a libmp3lame -q:a 3 "
        f"'{out_mp3}'"
    )
    os.system(ff)

    return send_file(out_mp3, mimetype='audio/mpeg', as_attachment=True, download_name=f'{speaker}_speech.mp3')

if __name__ == '__main__':
    # Railway provides PORT env var
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)


