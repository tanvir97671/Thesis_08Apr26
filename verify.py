"""Quick functional verification script — run before deploying to Lightning AI."""
import torch
import numpy as np

print("=" * 50)
print("Functional Verification")
print("=" * 50)

# Test 1: Twisted conv gradcheck
print("\n[1] Twisted conv gradcheck...")
from src.twisted_conv import twisted_conv, build_chirp_phase, twisted_conv_verify_noncommutativity
chirp = build_chirp_phase(4, 4, dtype=torch.complex128)
x = torch.randn(4, 4, dtype=torch.complex128, requires_grad=True)
g = torch.randn(4, 4, dtype=torch.complex128, requires_grad=True)
passed = torch.autograd.gradcheck(
    lambda a, b: twisted_conv(a, b, chirp), (x, g),
    eps=1e-6, atol=1e-4, rtol=1e-3
)
print(f"  Gradcheck: {passed}")

# Test 2: Noncommutativity
print("\n[2] Noncommutativity verification...")
ratio = twisted_conv_verify_noncommutativity(8, 8)
print(f"  Ratio: {ratio:.4f} (should be > 0.01)")
assert ratio > 0.01, "Noncommutativity test FAILED!"

# Test 3: NCTNet forward+backward
print("\n[3] NCTNet forward+backward...")
from src.nctnet import NCTNet
model = NCTNet(
    n_tau=8, n_nu=8, n_layers=2, hidden_dim=16,
    n_classes=4, n_paths=4, comm_head_dim=16, sens_head_dim=8, dropout=0
)
y = torch.randn(2, 8, 8, dtype=torch.complex64)
out = model(y)
loss = out["comm_logits"].float().sum() + out["sensing"].sum()
loss.backward()
print(f"  comm_logits: {out['comm_logits'].shape}")
print(f"  sensing: {out['sensing'].shape}")
print(f"  Params: {model.count_parameters():,}")

# Test 4: Dataset
print("\n[4] Dataset generation...")
from src.data.dataset import OTFSISACDataset
ds = OTFSISACDataset(
    n_samples=10, n_tau=8, n_nu=8, n_paths=4,
    max_delay_idx=2, max_doppler_idx=1, snr_db=10, seed=42
)
s = ds[0]
assert not torch.isnan(s["y_dd"]).any(), "NaN in dataset!"
print(f"  OK: {len(ds)} samples, shape {s['y_dd'].shape}, no NaN")

# Test 5: CNN baseline
print("\n[5] CNN baseline...")
from src.baselines.dl import CNNReceiver
cnn = CNNReceiver(n_tau=8, n_nu=8, n_classes=4, n_paths=4, hidden_dim=16, n_layers=2)
out_c = cnn(y)
print(f"  OK: {out_c['comm_logits'].shape}, params={cnn.count_parameters():,}")

# Test 6: Transformer baseline
print("\n[6] Transformer baseline...")
from src.baselines.dl import TransformerReceiver
tfm = TransformerReceiver(n_tau=8, n_nu=8, n_classes=4, n_paths=4, d_model=16, n_heads=2, n_encoder_layers=2)
out_t = tfm(y)
print(f"  OK: {out_t['comm_logits'].shape}, params={tfm.count_parameters():,}")

# Test 7: MMSE baseline
print("\n[7] MMSE baseline...")
from src.baselines.classical import MMSEDetector
mmse = MMSEDetector(n_tau=8, n_nu=8, n_classes=4)
h = torch.randn(2, 8, 8, dtype=torch.complex64) * 0.1
out_m = mmse(y, h, snr_db=10.0)
print(f"  OK: {out_m['comm_logits'].shape}")

# Test 8: Metrics
print("\n[8] Metrics...")
from src.utils.metrics import compute_ber, compute_ser
targets = torch.randint(0, 4, (2, 8, 8))
ber = compute_ber(out["comm_logits"].float(), targets)
ser = compute_ser(out["comm_logits"].float(), targets)
print(f"  BER: {ber:.4f}, SER: {ser:.4f}")

# Test 9: Mini training loop
print("\n[9] Mini training loop (2 epochs, 10 samples)...")
loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(2):
    for batch in loader:
        optimizer.zero_grad()
        o = model(batch["y_dd"])
        l = loss_fn(o["comm_logits"], batch["symbol_indices"].long())
        l.backward()
        optimizer.step()
    print(f"  Epoch {epoch+1}: loss={l.item():.4f}")

print("\n" + "=" * 50)
print("ALL FUNCTIONAL TESTS PASSED")
print("=" * 50)
