#!/usr/bin/env python3
import os
import tempfile
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import time, threading, uuid
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Simple in-memory metrics
_metrics_lock = threading.Lock()
_metrics = {
    'requests_total': 0,
    'requests_inflight': 0,
    'requests_by_speaker': defaultdict(int),
    'errors_total': 0,
    'latency_ms_sum': 0.0,
    'latency_ms_count': 0,
    'warmed': False,
    'started_at': time.time(),
}

def _metric_inc(key, amount=1):
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0) + amount

def _metric_obs_latency(ms):
    with _metrics_lock:
        _metrics['latency_ms_sum'] += float(ms)
        _metrics['latency_ms_count'] += 1

def _metric_set_warmed(val=True):
    with _metrics_lock:
        _metrics['warmed'] = bool(val)

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
        with _metrics_lock:
            warmed = _metrics.get('warmed', False)
            started_at = _metrics.get('started_at', 0)
        return jsonify({ 'status': 'ready' if ready else 'initializing', 'engine': 'xtts_v2', 'warmed': warmed, 'uptime_s': int(time.time()-started_at) })
    except Exception:
        return jsonify({ 'status': 'initializing', 'engine': 'xtts_v2' })

@app.post('/api/tts')
def tts_api():
    request_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    _metric_inc('requests_total')
    _metric_inc('requests_inflight')
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text') or ''
        speaker = (data.get('speaker') or data.get('speaker_id') or 'kelly').lower()
        language = data.get('language') or data.get('language_id') or 'en'
        if not text:
            _metric_inc('errors_total')
            return jsonify({ 'error': 'No text provided', 'request_id': request_id }), 400

        tts = ensure_model()

        refs = {
            'kelly': os.path.join(os.path.dirname(__file__), 'dist', 'reference_kelly.wav'),
            'ken': os.path.join(os.path.dirname(__file__), 'dist', 'reference_ken_mono16k.wav')
        }
        ref_wav = refs.get(speaker)
        if not (ref_wav and os.path.exists(ref_wav)):
            _metric_inc('errors_total')
            return jsonify({ 'error': f'reference wav not found for {speaker}', 'request_id': request_id }), 500

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

        with _metrics_lock:
            _metrics['requests_by_speaker'][speaker] += 1
        return send_file(out_mp3, mimetype='audio/mpeg', as_attachment=True, download_name=f'{speaker}_speech.mp3')
    except Exception as e:
        _metric_inc('errors_total')
        return jsonify({ 'error': str(e), 'request_id': request_id }), 500
    finally:
        dt_ms = (time.time() - t0) * 1000.0
        _metric_obs_latency(dt_ms)
        with _metrics_lock:
            _metrics['requests_inflight'] = max(0, _metrics['requests_inflight'] - 1)

if __name__ == '__main__':
    # Background model warm-up (non-blocking)
    def _warm_thread():
        try:
            t = ensure_model()
            # Light one-token warmup to initialize kernels
            refs = os.path.join(os.path.dirname(__file__), 'dist', 'reference_kelly.wav')
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
                try:
                    t.tts_to_file(text='Ready.', speaker_wav=refs, language='en', file_path=tmp.name)
                except Exception:
                    pass
            _metric_set_warmed(True)
        except Exception:
            pass
    threading.Thread(target=_warm_thread, daemon=True).start()

    # Railway provides PORT env var
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)

@app.get('/metrics')
def metrics():
    with _metrics_lock:
        lines = []
        lines.append('# HELP ilh_tts_requests_total Total TTS requests')
        lines.append('# TYPE ilh_tts_requests_total counter')
        lines.append(f"ilh_tts_requests_total {int(_metrics.get('requests_total',0))}")
        lines.append('# HELP ilh_tts_errors_total Total TTS errors')
        lines.append('# TYPE ilh_tts_errors_total counter')
        lines.append(f"ilh_tts_errors_total {int(_metrics.get('errors_total',0))}")
        lines.append('# HELP ilh_tts_requests_inflight Inflight requests')
        lines.append('# TYPE ilh_tts_requests_inflight gauge')
        lines.append(f"ilh_tts_requests_inflight {int(_metrics.get('requests_inflight',0))}")
        for spk, val in _metrics.get('requests_by_speaker', {}).items():
            lines.append(f"ilh_tts_requests_by_speaker_total{{speaker=\"{spk}\"}} {int(val)}")
        s = float(_metrics.get('latency_ms_sum',0.0)); c = float(_metrics.get('latency_ms_count',0.0))
        avg = (s/c) if c else 0.0
        lines.append('# HELP ilh_tts_latency_ms_avg Average request latency (ms)')
        lines.append('# TYPE ilh_tts_latency_ms_avg gauge')
        lines.append(f"ilh_tts_latency_ms_avg {avg:.2f}")
    return Response('\n'.join(lines) + '\n', mimetype='text/plain')


