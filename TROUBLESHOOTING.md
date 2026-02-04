# 🔧 Installation Troubleshooting Guide

## Current Issue: Network Timeouts

You're experiencing network timeouts during `pip install`. This is a common issue with slow or unstable internet connections.

---

## ✅ Solutions (Try in Order)

### Solution 1: Increase Timeout (Recommended)

```powershell
.\.venv\bin\python.exe -m pip install --default-timeout=200 gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

**Why this works**: Gives pip more time to download packages from PyPI servers.

---

### Solution 2: Install Packages One by One

If the timeout persists, install packages individually:

```powershell
# Install packages one at a time
.\.venv\bin\python.exe -m pip install --default-timeout=200 gymnasium
.\.venv\bin\python.exe -m pip install --default-timeout=200 numpy
.\.venv\bin\python.exe -m pip install --default-timeout=200 pandas
.\.venv\bin\python.exe -m pip install --default-timeout=200 stable-baselines3
.\.venv\bin\python.exe -m pip install --default-timeout=200 matplotlib
.\.venv\bin\python.exe -m pip install --default-timeout=200 tensorboard
```

**Why this works**: Smaller downloads are less likely to timeout.

---

### Solution 3: Use a Different PyPI Mirror

If you're in Cameroon or Africa, try using a closer mirror:

```powershell
# Use Tsinghua mirror (China, but often faster globally)
.\.venv\bin\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gymnasium numpy pandas stable-baselines3 matplotlib tensorboard

# Or use Aliyun mirror
.\.venv\bin\python.exe -m pip install -i https://mirrors.aliyun.com/pypi/simple/ gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

**Why this works**: Different servers may have better connectivity to your location.

---

### Solution 4: Download Wheels Manually (Offline Installation)

If internet is very unstable:

1. **On a computer with good internet**, download wheel files:
   ```powershell
   pip download gymnasium numpy pandas stable-baselines3 matplotlib tensorboard -d packages/
   ```

2. **Transfer the `packages/` folder** to your current computer (USB, cloud, etc.)

3. **Install from local files**:
   ```powershell
   .\.venv\bin\python.exe -m pip install --no-index --find-links=packages/ gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
   ```

**Why this works**: No internet required during installation.

---

### Solution 5: Use Conda Instead of Pip (Alternative)

If pip continues to fail, try Anaconda/Miniconda:

1. **Install Miniconda** from: https://docs.conda.io/en/latest/miniconda.html

2. **Create environment**:
   ```powershell
   conda create -n travel_cost python=3.11
   conda activate travel_cost
   ```

3. **Install packages**:
   ```powershell
   conda install -c conda-forge gymnasium numpy pandas matplotlib tensorboard
   pip install stable-baselines3
   ```

**Why this works**: Conda has different package servers and may have better connectivity.

---

### Solution 6: Wait and Retry

Sometimes network issues are temporary:

1. **Wait 30-60 minutes** for network to stabilize
2. **Try during off-peak hours** (early morning or late night)
3. **Use a different network** (mobile hotspot, different WiFi)

---

## 🧪 Verify Installation

After successful installation, verify with:

```powershell
.\.venv\bin\python.exe -c "import gymnasium; import numpy; import stable_baselines3; import matplotlib; print('✅ All packages installed successfully!')"
```

**Expected output**:
```
✅ All packages installed successfully!
```

---

## 📊 Installation Status Check

Check which packages are already installed:

```powershell
.\.venv\bin\python.exe -m pip list
```

This shows all installed packages. Look for:
- ✅ gymnasium
- ✅ numpy
- ✅ pandas
- ✅ stable-baselines3
- ✅ matplotlib
- ✅ tensorboard

---

## 🚨 Common Error Messages

### Error: "Could not find a version that satisfies the requirement"
**Solution**: Update pip first:
```powershell
.\.venv\bin\python.exe -m pip install --upgrade pip
```

### Error: "No module named 'pip'"
**Solution**: Reinstall pip:
```powershell
.\.venv\bin\python.exe -m ensurepip --upgrade
```

### Error: "SSL: CERTIFICATE_VERIFY_FAILED"
**Solution**: Use trusted host:
```powershell
.\.venv\bin\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

### Error: "Permission denied"
**Solution**: Run PowerShell as Administrator or use `--user` flag:
```powershell
.\.venv\bin\python.exe -m pip install --user gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

---

## 🔄 Alternative: Use System Python (Not Recommended)

If virtual environment continues to fail, you can use system Python temporarily:

```powershell
# Check if packages are already in system Python
python -c "import gymnasium; import stable_baselines3; print('Already installed!')"

# If not, install to system
python -m pip install gymnasium numpy pandas stable-baselines3 matplotlib tensorboard

# Then run scripts with system Python
python train_agent.py
python demo.py
python evaluate_model.py
```

**⚠️ Warning**: This installs packages globally and may cause conflicts with other projects.

---

## 📞 Still Having Issues?

If none of the above solutions work:

1. **Check your internet connection**:
   ```powershell
   Test-Connection pypi.org -Count 4
   ```

2. **Check if PyPI is accessible**:
   - Visit https://pypi.org in your browser
   - If it doesn't load, there may be network restrictions

3. **Check firewall/proxy settings**:
   - Corporate networks may block PyPI
   - Try using a VPN or mobile hotspot

4. **Use a different computer**:
   - Install on a computer with better internet
   - Transfer the entire `.venv` folder

---

## ✅ Once Installation Succeeds

After successful installation, proceed with:

```powershell
# Option 1: Run everything automatically
.\run_all.ps1

# Option 2: Step by step
.\.venv\bin\python.exe train_agent.py
.\.venv\bin\python.exe demo.py
.\.venv\bin\python.exe evaluate_model.py
.\.venv\bin\python.exe predict.py
```

---

## 📝 Quick Reference

| Problem | Solution |
|---------|----------|
| Timeout errors | Increase timeout: `--default-timeout=200` |
| Slow download | Use mirror: `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| No internet | Download wheels manually, install offline |
| SSL errors | Use trusted host: `--trusted-host pypi.org` |
| Permission errors | Run as admin or use `--user` |

---

**Good luck! Once installed, you'll have an amazing RL travel cost predictor! 🚗💨**
