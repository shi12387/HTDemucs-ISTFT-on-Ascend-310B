import sys
from pathlib import Path
from support import ROOT,OPS,baseline,sha,require
baseline();p=Path(sys.argv[1]).resolve()
for _,stem in OPS:
    for rel in (f'op_host/{stem}.cpp',f'op_host/{stem}_tiling.h',f'op_kernel/{stem}.cpp'):
        require(sha(p/rel)==sha(ROOT/rel),'Generated source differs: '+rel)
    require((p/f'build_out/op_api/include/aclnn_{stem}.h').is_file(),'Missing generated ACL header: '+stem)
require((p/'build_out/op_api/lib/libcust_opapi.so').is_file(),'Custom library has not been built')
print('[PROJECT] five operators and fifteen source files verified')
