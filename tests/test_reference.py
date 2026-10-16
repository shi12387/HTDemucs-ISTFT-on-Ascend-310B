"""CPU-only analytical tests for inverse signs, denormalization and crop indexing."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_tests import unpack,numpy_ola,reference,make_case
from support import baseline,INPUT_SHAPE

class InverseTests(unittest.TestCase):
    def setUp(self):
        self.window=(.5-.5*np.cos(2*np.pi*np.arange(4096,dtype=np.float64)/4096)).astype(np.float32)
    def test_sources(self):baseline()
    def test_zero(self):
        x,s=make_case('zero',2026);self.assertEqual(np.count_nonzero(reference(x,s,self.window,'numpy')),0)
    def test_unpack_stats_edges_and_order(self):
        x=np.zeros(INPUT_SHAPE,np.float32);s=np.zeros(8,np.float32);s[:2]=(.25,.5)
        p=x.reshape(8,2,2048,336);p[7,0,31,10]=4;p[7,1,31,10]=-6
        z=unpack(x,s)
        self.assertEqual(z[7,31,12],np.complex64(2.25-2.75j))
        self.assertEqual(np.count_nonzero(z[:,:,:2]),0);self.assertEqual(np.count_nonzero(z[:,:,338:]),0)
        self.assertEqual(np.count_nonzero(z[:,2048]),0);self.assertEqual(np.count_nonzero(z[:,0].imag),0)
        self.assertEqual(z[0,31,12],np.complex64(.25+.25j))
    def test_single_frame_sine_sign_and_ola_crop(self):
        x=np.zeros(INPUT_SHAPE,np.float32);s=np.zeros(8,np.float32);s[1]=1
        x.reshape(8,2,2048,336)[3,1,5,0]=-32
        y=reference(x,s,self.window,'numpy')
        # Retained output n=0 corresponds to local sample 1536 in source frame 2.
        local=1536+np.arange(128);signal=np.sin(2*np.pi*5*local/4096)
        env=np.zeros(351232,np.float32)
        for f in range(340):env[f*1024:f*1024+4096]+=self.window*self.window
        expected=signal*self.window[local]/env[3584:3584+128]
        np.testing.assert_allclose(y[3,:128],expected,atol=2e-7,rtol=2e-6)
        self.assertEqual(np.count_nonzero(y[:3]),0);self.assertEqual(np.count_nonzero(y[4:]),0)
    def test_restore_case_identity(self):
        a,s=make_case('tones',2026);b,t=make_case('tones_restore',2026)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(s,t))

if __name__=='__main__':unittest.main()
