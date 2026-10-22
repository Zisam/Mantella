# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all
import os

def _tree(rel_dir: str, prefix: str):
    out = []
    for root, _, files in os.walk(rel_dir):
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(prefix, os.path.relpath(root, rel_dir))
            out.append((src, dst))
    return out

datas = []
binaries = []
hiddenimports = ['tiktoken_ext.openai_public']
datas += _tree('data', 'data') # runtime data: language_support.csv, actions/, game CSVs, sounds
datas += collect_data_files('gradio', include_py_files=True) # gradio reads its .py sources at runtime (generates .pyi caches), frozen build must ship them
datas += collect_data_files('gradio_client', include_py_files=True)
datas += collect_data_files('altair')
datas += collect_data_files('silero_vad_lite')
datas += collect_data_files('faster_whisper') # assets/silero_vad.onnx needed when transcribing with vad_filter=True
tmp_ret = collect_all('moonshine_onnx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Mantella',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Mantella.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Mantella',
)
