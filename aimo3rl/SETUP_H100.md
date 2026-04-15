# H100 GPU Setup Guide for Virtualized Environments

This guide outlines the steps to resolve the "Error 802: system not yet initialized" and "Fabric State: In Progress" issues commonly seen with H100 GPUs in PCIe passthrough VMs.

## 1. Operating System Requirements
*   **OS:** Ubuntu 24.04 LTS (Noble Numbat)
*   **Kernel:** 6.8 or newer (required for proper H100 BAR resource mapping)

## 2. Install NVIDIA Drivers (Open Kernel Modules)
Use the Open Kernel variant, which is the recommended standard for Hopper architecture on modern kernels.

```bash
sudo apt update
sudo apt purge -y nvidia* libnvidia*
sudo apt autoremove -y
sudo apt install -y nvidia-driver-570-server-open nvidia-fabricmanager-570
```

## 3. Apply Hardware Handshake Workarounds
Create a configuration file to explicitly disable NVLink features that cause hangs in PCIe-only virtualized environments.

```bash
# Create the fix configuration
echo 'options nvidia NVreg_EnableGpuFirmware=1 NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_NvLinkDisable=1 NVreg_RegistryDwords="RMConnectToCuda=1;RMFabricAutoConfig=0"' | sudo tee /etc/modprobe.d/nvidia-h100-fix.conf

# Enable modesetting
echo "options nvidia-drm modeset=1" | sudo tee /etc/modprobe.d/nvidia-drm.conf

# Update the boot image and reboot
sudo update-initramfs -u
sudo reboot
```

## 4. Software Environment Setup
Use `uv` to manage dependencies and ensure binary compatibility.

```bash
# Sync environment
uv sync --reinstall

# Ensure CUDA 13.0 support for H100 (if using PyTorch nightly)
uv pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## 5. vLLM Optimization (Avoid v0.18.0 Compilation Bug)
When running `vLLM` version 0.18.0 with `torch` 2.11.0, you must bypass the experimental compilation backend to avoid an `AttributeError`.

Add the `--enforce-eager` flag to your server startup command:
```python
# In main.py or your launch script:
cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "openai/gpt-oss-20b",
    "--enforce-eager",  # <--- Crucial flag
    ...
]
```

## 6. Verification
After rebooting and setting up the environment, verify with:
```bash
nvidia-smi
# Fabric State should be "N/A" (instead of "In Progress")
# CUDA available should be "True"
python3 -c "import torch; print(torch.cuda.is_available())"
```
