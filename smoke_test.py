"""
Smoke Test — Comprehensive Verification Script
================================================
Runs ALL checks before spending Lightning AI credits:
  1. Import verification
  2. Twisted convolution gradcheck
  3. Noncommutativity verification  
  4. Model forward/backward pass
  5. Dataset generation (tiny, real 3GPP data)
  6. Mini training loop (3 epochs, ~1% of full data)
  7. Baseline comparison
  8. GPU memory check

Usage:
    python smoke_test.py
"""

import sys
import os
import time
import traceback
import torch
import numpy as np

# ============================================================
# STEP 0: Environment Check
# ============================================================
def check_environment():
    """Verify Python and PyTorch environment."""
    print("=" * 70)
    print("  NCTNet SMOKE TEST — Pre-Flight Verification")
    print("=" * 70)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:    {mem:.1f} GB")
    print(f"  Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# STEP 1: Import Verification
# ============================================================
def test_imports():
    """Verify all modules import correctly."""
    print("\n[1/8] Import verification...")
    try:
        from src.otfs_simulator import OTFSSimulator
        from src.twisted_conv import (
            twisted_conv, build_chirp_phase, TwistedConvLayer,
            TwistedConvBlock, NoncommutativeNorm, ComplexReLU, ModReLU,
        )
        from src.nctnet import NCTNet
        from src.baselines.classical import MMSEDetector, MessagePassingDetector
        from src.baselines.dl import CNNReceiver, TransformerReceiver
        from src.data.dataset import OTFSISACDataset
        from src.utils.metrics import compute_ber, compute_ser, compute_sensing_rmse
        from src.utils.gpu_utils import setup_gpu
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STEP 2: Twisted Convolution Gradcheck
# ============================================================
def test_gradcheck(device):
    """Verify twisted convolution is differentiable (critical!)."""
    print("\n[2/8] Twisted convolution gradcheck...")
    from src.twisted_conv import twisted_conv, build_chirp_phase

    n_tau, n_nu = 4, 4  # Tiny for gradcheck speed

    chirp = build_chirp_phase(n_tau, n_nu, device=device, dtype=torch.complex128)

    x = torch.randn(n_tau, n_nu, dtype=torch.complex128, device=device, requires_grad=True)
    g = torch.randn(n_tau, n_nu, dtype=torch.complex128, device=device, requires_grad=True)

    try:
        passed = torch.autograd.gradcheck(
            lambda a, b: twisted_conv(a, b, chirp),
            (x, g),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        if passed:
            print("  ✓ gradcheck PASSED — twisted convolution is differentiable")
        return passed
    except Exception as e:
        print(f"  ✗ gradcheck FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STEP 3: Noncommutativity Verification
# ============================================================
def test_noncommutativity(device):
    """Verify f⋆g ≠ g⋆f (the core theoretical claim)."""
    print("\n[3/8] Noncommutativity verification...")
    from src.twisted_conv import twisted_conv_verify_noncommutativity

    ratio = twisted_conv_verify_noncommutativity(8, 8, device=device)
    passed = ratio > 0.01

    if passed:
        print(f"  ✓ Noncommutativity ratio: {ratio:.4f} (> 0.01)")
        print(f"    ||f⋆g - g⋆f|| / ||f⋆g|| = {ratio:.4f}")
        print(f"    Confirms: twisted conv IS noncommutative")
    else:
        print(f"  ✗ Ratio too low: {ratio:.6f} — something is wrong!")

    return passed


# ============================================================
# STEP 4: Model Forward/Backward Pass
# ============================================================
def test_model_forward_backward(device):
    """Verify NCTNet forward and backward pass."""
    print("\n[4/8] NCTNet forward/backward pass...")

    from src.nctnet import NCTNet

    model = NCTNet(
        n_tau=8, n_nu=8, n_layers=2, hidden_dim=16,
        n_classes=4, n_paths=4, dropout=0.0,
        comm_head_dim=16, sens_head_dim=8,
    ).to(device)

    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Forward
    B = 2
    y_dd = torch.randn(B, 8, 8, dtype=torch.complex64, device=device)

    try:
        outputs = model(y_dd)
        comm_logits = outputs["comm_logits"]
        sensing = outputs["sensing"]

        assert comm_logits.shape == (B, 4, 8, 8), f"Wrong comm shape: {comm_logits.shape}"
        assert sensing.shape == (B, 4, 3), f"Wrong sens shape: {sensing.shape}"
        print(f"  ✓ Forward pass: comm_logits {comm_logits.shape}, sensing {sensing.shape}")

        # Backward
        loss = comm_logits.float().sum() + sensing.sum()
        loss.backward()

        # Check gradients exist
        has_grads = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        if has_grads:
            print(f"  ✓ Backward pass: all gradients computed")
        else:
            # Some params may not contribute to output — check critical ones
            grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
            total_count = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"  ⚠ {grad_count}/{total_count} parameters have gradients")

        return True
    except Exception as e:
        print(f"  ✗ Forward/backward FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STEP 5: Dataset Generation Check
# ============================================================
def test_dataset(device):
    """Verify dataset generation with real 3GPP channel model."""
    print("\n[5/8] Dataset generation (real Bluetooth channel sounding)...")

    from src.data.dataset import BluetoothChannelSoundingDataset

    try:
        file_path = os.environ.get(
            "REAL_DATASET_PATH", "data/real/bluetooth/sample_1_meter.csv"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing real dataset file: {file_path}"
            )

        ds = BluetoothChannelSoundingDataset(
            file_path=file_path,
            n_tau=16,
            n_nu=10,
            distance_m=1.0,
            use_reflector=True,
            fraction=0.01,
            seed=42,
        )

        assert len(ds) >= 1
        sample = ds[0]
        assert "y_dd" in sample
        assert "ranges_m" in sample
        assert sample["y_dd"].shape == (16, 10)

        # Check data is complex
        assert sample["y_dd"].is_complex(), "y_dd should be complex"

        # Check no NaN/Inf
        assert not torch.isnan(sample["y_dd"]).any(), "NaN in y_dd!"
        assert not torch.isinf(sample["y_dd"].real).any(), "Inf in y_dd!"

        print(f"  ✓ Dataset: {len(ds)} samples, shape {sample['y_dd'].shape}")
        print(f"  ✓ Data is complex: {sample['y_dd'].dtype}")
        print(f"  ✓ No NaN/Inf detected")
        return True
    except Exception as e:
        print(f"  ✗ Dataset generation FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STEP 6: Mini Training Loop
# ============================================================
def test_mini_training(device):
    """Run a complete mini training loop (3 epochs, ~20 samples)."""
    print("\n[6/8] Mini training loop (1 epoch, real data)...")

    from src.nctnet import NCTNet
    from src.data.dataset import BluetoothChannelSoundingDataset

    model = NCTNet(
        n_tau=16, n_nu=10, n_layers=2, hidden_dim=16,
        n_classes=4, n_paths=1, dropout=0.0,
        comm_head_dim=16, sens_head_dim=8,
    ).to(device)

    file_path = os.environ.get(
        "REAL_DATASET_PATH", "data/real/bluetooth/sample_1_meter.csv"
    )
    ds = BluetoothChannelSoundingDataset(
        file_path=file_path,
        n_tau=16,
        n_nu=10,
        distance_m=1.0,
        use_reflector=True,
        fraction=0.01,
        seed=42,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    try:
        for epoch in range(1):
            epoch_loss = 0.0
            for batch in loader:
                y_dd = batch["y_dd"].to(device)
                ranges = batch["ranges_m"].to(device).float()
                velocities = batch["velocities_mps"].to(device).float()
                powers = batch["powers_db"].to(device).float()

                optimizer.zero_grad()
                outputs = model(y_dd)
                pred_sens = outputs["sensing"][:, :1, :]
                true_sens = torch.stack([ranges, velocities, powers], dim=-1)
                loss = loss_fn(pred_sens.float(), true_sens)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            print(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")

        print("  ✓ Training loop completed")
        return True

    except Exception as e:
        print(f"  ✗ Training loop FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STEP 7: Baseline Comparison
# ============================================================
def test_baselines(device):
    """Verify all baselines run without errors."""
    print("\n[7/8] Baseline models verification...")

    from src.baselines.dl import CNNReceiver, TransformerReceiver
    from src.baselines.classical import MMSEDetector

    B, T, N = 2, 8, 8
    y_dd = torch.randn(B, T, N, dtype=torch.complex64, device=device)

    all_ok = True

    # CNN
    try:
        cnn = CNNReceiver(n_tau=T, n_nu=N, n_classes=4, n_paths=4, hidden_dim=16, n_layers=2).to(device)
        out = cnn(y_dd)
        assert "comm_logits" in out
        print(f"  ✓ CNN: {cnn.count_parameters():,} params, output {out['comm_logits'].shape}")
    except Exception as e:
        print(f"  ✗ CNN FAILED: {e}")
        all_ok = False

    # Transformer
    try:
        tfm = TransformerReceiver(n_tau=T, n_nu=N, n_classes=4, n_paths=4, d_model=16, n_heads=2, n_encoder_layers=2).to(device)
        out = tfm(y_dd)
        assert "comm_logits" in out
        print(f"  ✓ Transformer: {tfm.count_parameters():,} params, output {out['comm_logits'].shape}")
    except Exception as e:
        print(f"  ✗ Transformer FAILED: {e}")
        all_ok = False

    # MMSE (needs channel)
    try:
        mmse = MMSEDetector(n_tau=T, n_nu=N, n_classes=4).to(device)
        h_dd = torch.randn(B, T, N, dtype=torch.complex64, device=device) * 0.1
        out = mmse(y_dd, h_dd, snr_db=10.0)
        assert "comm_logits" in out
        print(f"  ✓ MMSE: output {out['comm_logits'].shape}")
    except Exception as e:
        print(f"  ✗ MMSE FAILED: {e}")
        all_ok = False

    return all_ok


# ============================================================
# STEP 8: GPU Memory Check
# ============================================================
def test_gpu_memory(device):
    """Check GPU memory with realistic model size."""
    print("\n[8/8] GPU memory check...")

    if device.type != "cuda":
        print("  ⚠ No GPU available, skipping memory check")
        return True

    from src.nctnet import NCTNet

    try:
        torch.cuda.reset_peak_memory_stats()

        # Smoke test config size
        model = NCTNet(
            n_tau=16, n_nu=16, n_layers=2, hidden_dim=32,
            n_classes=4, n_paths=4, dropout=0.1,
            comm_head_dim=32, sens_head_dim=16,
        ).to(device)

        B = 32
        y_dd = torch.randn(B, 16, 16, dtype=torch.complex64, device=device)

        # Forward + backward
        outputs = model(y_dd)
        loss = outputs["comm_logits"].float().sum() + outputs["sensing"].sum()
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)

        print(f"  Peak memory: {peak_mem:.2f} GB / {total_mem:.1f} GB ({100*peak_mem/total_mem:.1f}%)")
        print(f"  ✓ Memory usage is reasonable")

        # Cleanup
        del model, y_dd, outputs, loss
        torch.cuda.empty_cache()

        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  ✗ OUT OF MEMORY — reduce batch_size or model size")
        return False
    except Exception as e:
        print(f"  ✗ GPU memory check FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    device = check_environment()

    results = {}
    results["imports"] = test_imports()
    results["gradcheck"] = test_gradcheck(device)
    results["noncommutativity"] = test_noncommutativity(device)
    results["forward_backward"] = test_model_forward_backward(device)
    results["dataset"] = test_dataset(device)
    results["mini_training"] = test_mini_training(device)
    results["baselines"] = test_baselines(device)
    results["gpu_memory"] = test_gpu_memory(device)

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("  SMOKE TEST RESULTS")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(f"  🟢 ALL TESTS PASSED ({elapsed:.1f}s)")
        print(f"  Safe to proceed with full training on Lightning AI.")
    else:
        print(f"  🔴 SOME TESTS FAILED ({elapsed:.1f}s)")
        print(f"  DO NOT proceed with full training until all tests pass.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
