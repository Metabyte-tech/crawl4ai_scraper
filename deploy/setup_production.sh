#!/bin/bash

# setup_production.sh - Helper script to install systemd services for Crawl4AI
# This script will automatically detect your current path and set up the services.

# Get the absolute path to the directory where this script resides, then go up one level
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)

echo "--- Installing Crawl4AI Systemd Services ---"
echo "Detected Project Directory: $PROJECT_DIR"
echo "Detected User: $CURRENT_USER"

# Generate actual service files with the correct paths dynamically
cat > "$SCRIPT_DIR/crawl4ai-api.service" << EOF
[Unit]
Description=Crawl4AI FastAPI Search Backend
After=network.target redis.service

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv_new/bin/python3 api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > "$SCRIPT_DIR/crawl4ai-worker.service" << EOF
[Unit]
Description=Crawl4AI Background Worker (arq)
After=network.target redis.service

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv_new/bin/arq worker.WorkerSettings
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Copy service files to systemd directory
echo "Step 1: Copying service files to systemd..."
sudo cp "$SCRIPT_DIR/crawl4ai-api.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/crawl4ai-worker.service" /etc/systemd/system/

# Reload systemd to recognize new units
echo "Step 2: Reloading systemd..."
sudo systemctl daemon-reload

# Enable and start services
echo "Step 3: Enabling and starting services..."
sudo systemctl enable crawl4ai-api
sudo systemctl enable crawl4ai-worker
sudo systemctl restart crawl4ai-api
sudo systemctl restart crawl4ai-worker

echo "--- Setup Complete ---"
echo "Check status:"
echo "  systemctl status crawl4ai-api"
echo "  systemctl status crawl4ai-worker"
