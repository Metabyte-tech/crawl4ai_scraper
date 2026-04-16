#!/bin/bash

# setup_production.sh - Helper script to install systemd services for Crawl4AI

PROJECT_DIR="/home/himanshu/workspace/ai-agent/crawl4AI"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "--- Installing Crawl4AI Systemd Services ---"

# Copy service files to systemd directory
echo "Step 1: Copying service files..."
sudo cp "$DEPLOY_DIR/crawl4ai-api.service" /etc/systemd/system/
sudo cp "$DEPLOY_DIR/crawl4ai-worker.service" /etc/systemd/system/

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
echo "Check logs:"
echo "  journalctl -u crawl4ai-worker -f"
