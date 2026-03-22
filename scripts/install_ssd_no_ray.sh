#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Convenience installer for the SSD environments (Harvest/Cleanup) WITHOUT Ray.
#
# The SSD reference repo pins ray[rllib]==0.8.5 which is not installable on
# modern systems (especially Apple Silicon).  This script:
#   1. Clones the repo (if needed)
#   2. Patches out ALL Ray/RLlib imports (replaced with lightweight stubs)
#   3. Fixes bare "import utility_funcs" (file lives at repo root, not in pkg)
#   4. Installs the package in editable mode with --no-deps
#
# IMPORTANT: SSD requires Python < 3.10.
# ============================================================================

REPO_URL="https://github.com/eugenevinitsky/sequential_social_dilemma_games"
TARGET_DIR="${1:-sequential_social_dilemma_games}"

# --- Python version guard ---
python3 - <<'PY'
import sys
maj, minor = sys.version_info[:2]
if (maj, minor) >= (3, 10):
    raise SystemExit(
        f"ERROR: SSD requires Python < 3.10, but you are on {sys.version.split()[0]}\n"
        "Create a Python 3.9 environment (e.g., conda create -n imlssd python=3.9) and rerun."
    )
print(f"[install_ssd_no_ray] Python OK: {sys.version.split()[0]}")
PY

# --- Cleanup potentially broken installs ---
echo "[install_ssd_no_ray] Cleaning any previous installs (safe if not present)..."
python3 -m pip uninstall -y social-dilemmas social_dilemmas ray rllib dm_tree tree 2>/dev/null || true
python3 -m pip install --upgrade pip setuptools wheel

# --- Clone if needed ---
if [[ -d "$TARGET_DIR" ]]; then
    echo "[install_ssd_no_ray] Target directory already exists: $TARGET_DIR"
else
    git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# --- Install deps (excluding Ray) ---
echo "[install_ssd_no_ray] Installing dependencies (excluding Ray)..."
if [[ -f requirements.txt ]]; then
    grep -iv '^ray' requirements.txt > requirements_no_ray.txt || true
    python3 -m pip install -r requirements_no_ray.txt 2>/dev/null || {
        echo "[install_ssd_no_ray] Some deps failed; installing minimal set..."
        python3 -m pip install "gym>=0.21.0" "numpy<2" "opencv-python<4.13" scipy matplotlib
    }
else
    python3 -m pip install "gym>=0.21.0" "numpy<2" "opencv-python<4.13" scipy matplotlib
fi

# ============================================================================
# PATCH 1: Copy utility_funcs.py into the social_dilemmas package
# (It lives at the repo root but is imported as "import utility_funcs")
# ============================================================================
echo "[install_ssd_no_ray] Patching utility_funcs import..."
if [[ -f utility_funcs.py ]]; then
    cp utility_funcs.py social_dilemmas/utility_funcs.py
    echo "  Copied utility_funcs.py -> social_dilemmas/utility_funcs.py"
fi

# Fix the import in agent.py (and anywhere else that does bare import)
# Original:  import utility_funcs as util
# Patched:   from social_dilemmas import utility_funcs as util
find social_dilemmas/ -name "*.py" ! -name "*.bak" -exec sed -i.bak \
    's/^import utility_funcs/from social_dilemmas import utility_funcs/g' {} +
find social_dilemmas/ -name "*.py" ! -name "*.bak" -exec sed -i.bak \
    's/^from utility_funcs/from social_dilemmas.utility_funcs/g' {} +

# ============================================================================
# PATCH 2: Stub out ALL Ray/RLlib imports across the entire package.
#
# map_env.py imports:
#   from ray.rllib.agents.callbacks import DefaultCallbacks
#   from ray.rllib.env import MultiAgentEnv
#
# We replace these with lightweight stubs so the envs work without Ray.
# ============================================================================
echo "[install_ssd_no_ray] Patching Ray/RLlib imports..."

# Use a single Python script to patch ALL .py files in social_dilemmas/
python3 - <<'PYFIX'
import os, re, glob

pkg_dir = "social_dilemmas"

for pyfile in glob.glob(os.path.join(pkg_dir, "**", "*.py"), recursive=True):
    if pyfile.endswith(".bak"):
        continue

    with open(pyfile, "r") as f:
        original = f.read()

    content = original
    lines = content.split("\n")
    new_lines = []
    changed = False

    for line in lines:
        stripped = line.strip()

        # Skip already-patched lines
        if "# patched by install_ssd_no_ray" in line:
            new_lines.append(line)
            continue

        # Match: from ray.rllib.agents.callbacks import DefaultCallbacks
        if re.match(r"^\s*from\s+ray\.rllib\.agents\.callbacks\s+import\s+DefaultCallbacks", stripped):
            new_lines.append("# --- Ray stub (patched by install_ssd_no_ray.sh) ---")
            new_lines.append("try:")
            new_lines.append("    from ray.rllib.agents.callbacks import DefaultCallbacks")
            new_lines.append("except (ImportError, ModuleNotFoundError):")
            new_lines.append("    class DefaultCallbacks:")
            new_lines.append("        pass")
            changed = True

        # Match: from ray.rllib.env import MultiAgentEnv
        elif re.match(r"^\s*from\s+ray\.rllib\.env\s+import\s+MultiAgentEnv", stripped):
            new_lines.append("try:")
            new_lines.append("    from ray.rllib.env import MultiAgentEnv")
            new_lines.append("except (ImportError, ModuleNotFoundError):")
            new_lines.append("    import gym")
            new_lines.append("    class MultiAgentEnv(gym.Env):")
            new_lines.append('        """Minimal stub for ray.rllib.env.MultiAgentEnv."""')
            new_lines.append("        pass")
            new_lines.append("# --- End Ray stub ---")
            changed = True

        # Match any other ray import
        elif re.match(r"^\s*(from\s+ray\.|import\s+ray)", stripped):
            new_lines.append("try:  # patched by install_ssd_no_ray.sh")
            new_lines.append("    " + line)
            new_lines.append("except (ImportError, ModuleNotFoundError):")
            new_lines.append("    pass")
            changed = True

        else:
            new_lines.append(line)

    if changed:
        content = "\n".join(new_lines)
        with open(pyfile, "w") as f:
            f.write(content)
        print(f"  Patched: {pyfile}")

print("  Ray/RLlib patching complete.")
PYFIX

# Clean up .bak files
find social_dilemmas/ -name "*.bak" -delete 2>/dev/null || true

# --- Editable install (no deps to avoid Ray requirement) ---
echo "[install_ssd_no_ray] Installing social-dilemmas (editable, no deps)..."
python3 -m pip install --no-deps -e .

# --- Verify ---
echo ""
echo "[install_ssd_no_ray] Verifying installation..."
python3 -c "
from social_dilemmas.envs.cleanup import CleanupEnv
env = CleanupEnv(num_agents=5)
obs = env.reset()
print(f'  CleanupEnv OK: {len(obs)} agents, obs shape = {list(obs.values())[0].shape}')
from social_dilemmas.envs.harvest import HarvestEnv
env2 = HarvestEnv(num_agents=5)
obs2 = env2.reset()
print(f'  HarvestEnv OK: {len(obs2)} agents, obs shape = {list(obs2.values())[0].shape}')
print()
print('=== SSD installation successful! ===')
"
