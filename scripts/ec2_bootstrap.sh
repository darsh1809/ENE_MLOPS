#!/usr/bin/env bash
# ============================================================
# scripts/ec2_bootstrap.sh
# Run ONCE on a fresh Amazon Linux 2023 / Ubuntu EC2 instance
# to install Docker and prepare it for the CI/CD deployment.
#
# Usage (from your local machine):
#   scp scripts/ec2_bootstrap.sh ec2-user@<EC2_HOST>:~/
#   ssh ec2-user@<EC2_HOST> "bash ~/ec2_bootstrap.sh"
# ============================================================
set -euo pipefail

echo "=============================="
echo "  EC2 Bootstrap — MLOps Setup"
echo "=============================="

# ── Detect OS ─────────────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS." && exit 1
fi

# ── Install Docker ─────────────────────────────────────────────────────────
echo "[1/4] Installing Docker..."

if [[ "$OS" == "amzn" ]]; then
    # Amazon Linux 2023
    sudo dnf update -y
    sudo dnf install -y docker git
elif [[ "$OS" == "ubuntu" ]]; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io git
else
    echo "Unsupported OS: $OS" && exit 1
fi

# ── Start & enable Docker ──────────────────────────────────────────────────
echo "[2/4] Starting Docker daemon..."
sudo systemctl start docker
sudo systemctl enable docker

# ── Allow current user to run Docker without sudo ─────────────────────────
echo "[3/4] Adding user '$USER' to docker group..."
sudo usermod -aG docker "$USER"

# ── Create app directory ───────────────────────────────────────────────────
echo "[4/4] Creating /home/$USER/app directory..."
mkdir -p "/home/$USER/app/models"

echo ""
echo "=============================="
echo "  Bootstrap complete!"
echo "  NOTE: Log out and back in for docker group to take effect."
echo "  Then the GitHub Actions workflow can deploy directly."
echo "=============================="
