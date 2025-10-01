set -eo pipefail

DEFAULT_SPACE_NAME="Arabic-Diacritizer-Demo"
STAGING_DIR="_hf_space_staging"

usage() {
  cat <<EOF
🚀 Hugging Face Space Deployment Script

This script builds and deploys the Gradio demo to a Hugging Face Space.
It builds the 'common' and 'inference' packages into wheels and pushes them
along with the application code.

Usage:
  ./deploy_space.sh [OPTIONS]

Options:
  -u, --username   Hugging Face username.
                   (Overrides HF_USERNAME environment variable)
  -s, --space      The name for the new Space.
                   (Default: "$DEFAULT_SPACE_NAME")
  -t, --token      Hugging Face access token with 'write' permissions.
                   (Overrides HF_TOKEN environment variable. Use with caution.)
  -h, --help       Display this help message and exit.

Configuration Hierarchy (Highest to Lowest):
1. Command-Line Arguments (e.g., --username)
2. Environment Variables (e.g., export HF_USERNAME=...)
3. Interactive Prompts (for username and token)
EOF
  exit 1
}

HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
SPACE_NAME="$DEFAULT_SPACE_NAME"

# This loop will override the values from environment variables if arguments are provided.
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -u|--username)
      HF_USERNAME="$2"
      shift 2
      ;;
    -s|--space)
      SPACE_NAME="$2"
      shift 2
      ;;
    -t|--token)
      HF_TOKEN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "❌ Unknown parameter passed: $1"
      usage
      ;;
  esac
done

echo "🚀 Starting Hugging Face Space deployment..."

if [ -z "$HF_USERNAME" ]; then
  read -p "Enter your Hugging Face username: " HF_USERNAME
  if [ -z "$HF_USERNAME" ]; then
    echo "❌ Username is required. Exiting."
    exit 1
  fi
fi

if [ -z "$HF_TOKEN" ]; then
  echo "🔑 Please paste your Hugging Face User Access Token (with write permissions):"
  read -s HF_TOKEN
  if [ -z "$HF_TOKEN" ]; then
    echo "❌ Access token is required. Exiting."
    exit 1
  fi
fi

echo "----------------------------------------"
echo "Deployment Configuration:"
echo "  - Hugging Face Username: $HF_USERNAME"
echo "  - Space Name:            $SPACE_NAME"
echo "----------------------------------------"

echo "🔐 Authenticating with Hugging Face Hub..."
hf auth login --token "$HF_TOKEN" --add-to-git-credential > /dev/null 2>&1
echo "✅ Authentication successful."

echo "🧹 Cleaning up and creating a fresh staging directory: '$STAGING_DIR'"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# trap 'echo "❗️ An error occurred. Cleaning up..."; rm -rf "$STAGING_DIR"' ERR INT TERM

echo "📦 Copying 'common' and 'inference' source code..."
# Create the target directories in the staging area
mkdir -p "$STAGING_DIR/arabic_diacritizer_common"
mkdir -p "$STAGING_DIR/diacritizer"
# Copy the contents of the src directories
cp -r common/src/arabic_diacritizer_common/* "$STAGING_DIR/arabic_diacritizer_common/"
cp -r inference/src/diacritizer/* "$STAGING_DIR/diacritizer/"
echo "✅ Source code copied."

echo "🚚 Assembling application files..."
cp demo_app/app.py "$STAGING_DIR/"

echo "📝 Creating requirements.txt for the Space..."
cat <<EOL > "$STAGING_DIR/requirements.txt"
# Gradio for the UI
gradio>=4.0.0

# Core dependencies
onnxruntime
numpy
huggingface-hub

# --- Custom Local Packages ---
# These will be installed from the .whl files in the repo
# arabic_diacritizer_common
# arabic_diacritizer
EOL

echo "📖 Creating README.md with Space metadata..."
cat <<EOL > "$STAGING_DIR/README.md"
---
title: Arabic Diacritizer Demo
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.47.2
app_file: app.py
pinned: false
license: mit
---
EOL
echo "✅ Assembly complete."

echo "☁️ Creating and deploying the Space: '$HF_USERNAME/$SPACE_NAME'"
hf repo create $HF_USERNAME/$SPACE_NAME --repo-type space --space_sdk gradio 
# > /dev/null 2>&1
# --space-sdk gradio

AUTHENTICATED_REPO_URL="https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "🔄 Cloning the new Space repository (this may take a moment)..."
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" _repo/

cp -r "$STAGING_DIR"/* _repo/

cd _repo

echo "🚀 Pushing files to the Space repository (this may be slow)..."
git add .
if git diff-index --quiet HEAD; then
  echo "ℹ️ No changes to commit. Application is already up to date."
else
  git commit -m "Deploy application"
  git push
fi

echo ""
echo "🎉 Deployment successful!"
echo "Visit your new Space at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
