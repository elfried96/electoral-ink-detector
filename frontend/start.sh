#!/bin/bash

# Script de démarrage pour Next.js avec Node.js 20+
echo "Démarrage de l'application Next.js..."

# Charger nvm et utiliser Node 20
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Utiliser Node 20
nvm use 20

# Démarrer l'application
npm run dev