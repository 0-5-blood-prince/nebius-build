#!/usr/bin/env bash
# RoboMind setup script — run TONIGHT before the hackathon
set -e

echo "=== RoboMind Setup ==="

# 1. Install Python deps
echo "[1/4] Installing Python packages..."
pip3 install -r requirements.txt

# 2. Clone MuJoCo Menagerie (G1 model)
if [ ! -d "mujoco_menagerie" ]; then
  echo "[2/4] Cloning mujoco_menagerie..."
  git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie
else
  echo "[2/4] mujoco_menagerie already present, skipping."
fi

# 3. Verify G1 loads
echo "[3/4] Verifying G1 model loads..."
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/unitree_g1/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print('  G1 loaded! qpos shape:', m.nq, '  nu:', m.nu)
"

# 4. Verify Nebius API key
echo "[4/4] Verifying Nebius Token Factory connection..."
if [ -z "$NEBIUS_API_KEY" ]; then
  echo "  WARNING: NEBIUS_API_KEY not set. Export it before running demo.py:"
  echo "  export NEBIUS_API_KEY=your_key_here"
else
  python3 -c "
from openai import OpenAI
client = OpenAI(base_url='https://api.studio.nebius.com/v1/', api_key='$NEBIUS_API_KEY')
models = client.models.list()
print('  Connected! Available models:')
for m in list(models)[:5]:
    print('   -', m.id)
"
fi

echo ""
echo "=== Setup complete. Run: python3 demo.py ==="
