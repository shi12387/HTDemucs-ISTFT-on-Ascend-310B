import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
CASES=('tones_warmup','impulse','tones','noise','zero','tones_restore')
INPUT_SHAPE=(1,16,2048,336)
OUTPUT_SHAPE=(8,344064)
OPS=(('HtDecoderPackSplit','ht_decoder_pack_split'),('HtDftMatmul','ht_dft_matmul'),
     ('HtIstftMerge','ht_istft_merge'),('HtIstftWindowOla','ht_istft_window_ola'),
     ('HtIstftCompensateCrop','ht_istft_compensate_crop'))

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def info(path):
    p=Path(path);return {'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(path,item):
    p=Path(path);require(p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Missing/changed file: '+str(p))
def load(path,shape):
    a=np.fromfile(path,dtype='<f4');require(a.size==int(np.prod(shape)) and np.isfinite(a).all(),'Invalid tensor: '+str(path));return a.reshape(shape)
def baseline():
    m=json.loads((ROOT/'baseline_sources.json').read_text());require(len(m['files'])==15,'Expected fifteen operator source files')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Operator source changed: '+rel)
def sources():
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+[ROOT/'baseline_sources.json']
    for d in ('op_host','op_kernel','runtime','tests'):paths.extend((ROOT/d).glob('*'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}
def inputs():
    m=json.loads((ROOT/'test_data/manifest.json').read_text())
    require(m['cases']==list(CASES) and m['input_shape']==list(INPUT_SHAPE) and m['output_shape']==list(OUTPUT_SHAPE),'Changed test contract')
    require(m['contract']==contract(),'Changed inverse convention')
    for name,item in m['files'].items():checked(ROOT/'test_data'/name,item)
    for rel,digest in m['sources'].items():require(sha(ROOT/rel)==digest,'Source changed after preparation: '+rel)
    return m
def contract():
    return {'n_fft':4096,'hop_length':1024,'normalized':True,'periodic_hann':True,
            'packed_frames':336,'full_frames':340,'frame_insert':[2,338],
            'nyquist':0,'imag_dc_ignored':True,'crop_start':3584,'output_length':344064,
            'stats_mean_index':0,'stats_std_index':1,'stats_padding':[2,8]}
