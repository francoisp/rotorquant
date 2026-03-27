from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer
from .rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
from .clifford import geometric_product, make_random_rotor, rotor_sandwich

# Triton kernels (optional, requires triton >= 3.0)
try:
    from .triton_kernels import (
        triton_rotor_sandwich,
        triton_rotor_full_fused,
        triton_rotor_inverse_sandwich,
        triton_fused_attention,
        pack_rotors_for_triton,
    )
    _triton_available = True
except ImportError:
    _triton_available = False
