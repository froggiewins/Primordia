"""Embed models/primordia_parts.glb into src/primordia.src.html -> primordia.html.

Run after rebuilding the models (or after exporting the GLB from Blender by hand):
    python tools/embed_models.py
The result is a single self-contained HTML file you can double-click.
"""
import base64, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, 'src', 'primordia.src.html'), encoding='utf-8').read()
glb = open(os.path.join(ROOT, 'models', 'primordia_parts.glb'), 'rb').read()
marker = '__MODELS_B64__'
assert marker in src, 'marker missing from source'
out = src.replace(marker, base64.b64encode(glb).decode('ascii'))
open(os.path.join(ROOT, 'primordia.html'), 'w', encoding='utf-8').write(out)
print('wrote primordia.html (%d KB, models %d KB)' % (len(out) // 1024, len(glb) // 1024))
