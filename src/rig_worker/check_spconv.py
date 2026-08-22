"""Does spconv-cu120 actually work against a cu130 torch build?

spconv publishes no cu130 wheel. It bundles its own CUDA runtime via cumm, so
in theory it loads fine against a newer driver - but "in theory" is how the
torchmcubes CPU-only claim went wrong earlier in this project. This runs a real
sparse convolution on the GPU rather than trusting a successful import: a
mismatched CUDA runtime typically imports cleanly and fails at first kernel
launch.
"""
import torch
import spconv.pytorch as spconv

print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("spconv", spconv.__version__ if hasattr(spconv, "__version__") else "?")

# One real sparse conv on the GPU: 4 occupied voxels in a 1x8x8x8 grid.
feats = torch.randn(4, 3, device="cuda")
coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1],
                       [0, 2, 2, 2], [0, 3, 3, 3]],
                      dtype=torch.int32, device="cuda")
x = spconv.SparseConvTensor(feats, coords, [8, 8, 8], 1)
conv = spconv.SubMConv3d(3, 5, 3, bias=False, indice_key="t").cuda()
out = conv(x)
torch.cuda.synchronize()
print("sparse conv ran, out features:", tuple(out.features.shape))
print("SPCONV_OK")
