# VS Code ↔ Google Colab Connection Guide

## Method: Official Google Colab Extension (Recommended)

Google provides an **official VS Code extension** that lets you run notebooks locally while using Colab's cloud GPU. This is the safest, most reliable method.

> ⚠️ **Do NOT use** SSH tunneling methods (`colab-ssh`, `ngrok`, `cloudflared`). Google may flag or restrict your account.

---

## Setup Steps

### 1. Install the Extension

1. Open **VS Code**
2. Press `Cmd+Shift+X` to open Extensions
3. Search for **"Google Colab"** (published by **Google**)
4. Click **Install**
5. Also ensure the **Jupyter** extension is installed (it's a dependency)

### 2. Connect to a Colab Runtime

1. Open any `.ipynb` notebook file in VS Code (e.g., `notebooks/zonal_seg_colab.ipynb`)
2. Click **Select Kernel** (top-right corner of the notebook)
3. Choose **Colab** from the kernel options
4. Sign in with your Google account when prompted
5. Select your desired runtime:
   - **GPU** (recommended: A100 or G4)
   - Toggle **High RAM** ON

### 3. Workflow

Once connected:
- **Code editing** happens locally in VS Code (with IntelliSense, Git, etc.)
- **Execution** happens on Colab's GPU
- **Files on Colab** are accessed via `!` shell commands
- **Google Drive** is mounted via the notebook's first cell

---

## Benefits

| Feature | Browser Colab | VS Code + Colab Extension |
|---------|--------------|--------------------------|
| Code completion | Basic | Full IntelliSense |
| Git integration | Manual | Built-in |
| File navigation | Limited | Full project tree |
| Debugging | None | Full debugger |
| Terminal | None | Integrated |
| Extensions | None | All VS Code extensions |

---

## Alternative: Direct Browser

If the extension doesn't work for your setup, you can still:
1. Push code to GitHub from VS Code
2. Open `notebooks/zonal_seg_colab.ipynb` directly in [colab.research.google.com](https://colab.research.google.com)
3. The notebook's first cell will `git clone` your repo into Colab

Both workflows are fully supported by our pipeline.
