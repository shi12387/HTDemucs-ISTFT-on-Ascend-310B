#!/usr/bin/env python3
"""Generate packed spectra and independent CPU iSTFT references."""
import argparse
import json
import numpy as np
from support import ROOT,CASES,INPUT_SHAPE,OUTPUT_SHAPE,require,info,baseline,sources,contract

def unpack(x,stats):
    require(x.shape==INPUT_SHAPE and stats.shape==(8,),'Wrong input shape')
    # Match separate FP32 multiply/add operations before constructing complex data.
    planes=(x.reshape(8,2,2048,336)*np.float32(stats[1])).astype(np.float32)
    planes=(planes+np.float32(stats[0])).astype(np.float32)
    z=np.zeros((8,2049,340),np.complex64)
    z[:,:2048,2:338]=planes[:,0]+np.complex64(1j)*planes[:,1]
    z[:,0]=z[:,0].real
    return z

def numpy_ola(frames,window):
    require(frames.shape==(8,340,4096),'Wrong inverse frame shape')
    num=np.zeros((8,351232),np.float32);env=np.zeros(351232,np.float32)
    ww=(window*window).astype(np.float32)
    for frame in range(340):
        pos=frame*1024
        num[:,pos:pos+4096]+=(frames[:,frame]*window).astype(np.float32)
        env[pos:pos+4096]+=ww
    selected=env[3584:3584+344064]
    require(np.isfinite(selected).all() and np.min(selected)>1e-11,'Invalid selected window envelope')
    return np.ascontiguousarray(num[:,3584:3584+344064]/selected,dtype='<f4')

def reference(x,stats,window,backend):
    z=unpack(x,stats)
    if backend=='torch':
        import torch
        y=torch.istft(torch.from_numpy(z),n_fft=4096,hop_length=1024,win_length=4096,
                      window=torch.from_numpy(window),center=True,normalized=True,
                      onesided=True,length=347136,return_complex=False)
        y=y[:,1536:1536+344064].numpy()
    else:
        frames=np.fft.irfft(z.transpose(0,2,1).astype(np.complex128),n=4096,axis=-1,norm='ortho').astype(np.float32)
        y=numpy_ola(frames,window)
    require(y.shape==OUTPUT_SHAPE and np.isfinite(y).all(),'Invalid inverse reference')
    return np.ascontiguousarray(y,dtype='<f4')

def make_case(name,seed):
    x=np.zeros(INPUT_SHAPE,dtype='<f4');s=np.zeros(8,dtype='<f4');s[1]=1
    p=x.reshape(8,2,2048,336)
    if name.startswith('tones'):
        # Distinct channels, phases and interior frequencies expose ordering/sign errors.
        t=np.arange(336,dtype=np.float64)
        for c in range(8):
            phase=.13*c+.07*t
            p[c,0,5+17*c]=(.8+.1*c)*np.cos(phase)
            p[c,1,5+17*c]=(.8+.1*c)*np.sin(phase)
            p[c,0,2047-c]=.2*np.cos(.11*t+c)
            p[c,1,2047-c]=.2*np.sin(.11*t+c)
            p[c,0,0]=.03*(c+1)
    elif name=='impulse':
        k=np.arange(2048,dtype=np.float64)
        for c in range(8):
            phase=2*np.pi*k*(1003+23*c)/4096
            p[c,0,:,37+3*c]=(.1+.02*c)*np.cos(phase)
            p[c,1,:,37+3*c]=-(.1+.02*c)*np.sin(phase)
    elif name=='noise':
        rng=np.random.default_rng(seed);x[:]=(rng.standard_normal(INPUT_SHAPE)*.025).astype('<f4')
        s[0]=.003;s[1]=.73
    elif name!='zero':raise ValueError('Unknown case: '+name)
    return x,s

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',choices=('torch','numpy'),default='torch')
    parser.add_argument('--seed',type=int,default=2026);a=parser.parse_args();baseline()
    d=ROOT/'test_data';d.mkdir(exist_ok=True)
    if a.reference=='torch':
        import torch
        window=torch.hann_window(4096,periodic=True,dtype=torch.float32).numpy();version=torch.__version__
    else:
        window=(.5-.5*np.cos(2*np.pi*np.arange(4096,dtype=np.float64)/4096)).astype('<f4');version=np.__version__
    files={};window.tofile(d/'window_hann_f32.bin');files['window_hann_f32.bin']=info(d/'window_hann_f32.bin')
    cache=None
    for name in CASES:
        x,s=make_case(name,a.seed)
        require(np.isfinite(x).all() and np.max(np.abs(x))*abs(s[1])+abs(s[0])<65504,'Invalid or overflowing spectrum')
        if name.startswith('tones') and cache is not None:y=cache
        else:
            y=reference(x,s,window,a.reference)
            if name.startswith('tones'):cache=y
        for suffix,values in (('_input_f32.bin',x),('_stats_f32.bin',s),('_reference_f32.bin',y)):
            file=d/(name+suffix);values.tofile(file);files[file.name]=info(file)
        print('[PREPARED]',name,flush=True)
    m={'cases':list(CASES),'seed':a.seed,'reference_backend':a.reference,'reference_version':str(version),
       'input_shape':list(INPUT_SHAPE),'output_shape':list(OUTPUT_SHAPE),'contract':contract(),
       'files':files,'sources':sources()}
    (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print('[TEST DATA] six packed-spectrum cases, '+a.reference+' reference; no model assets required')

if __name__=='__main__':main()
