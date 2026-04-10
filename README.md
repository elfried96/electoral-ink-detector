# Ink Detection  Détection d'encre électorale

## Stack
Backend  : FastAPI + MediaPipe Tasks API + uv
Frontend : Next.js 14 + Tailwind CSS
Deploy   : Render (backend) + Vercel (frontend)

## Lancement local

Backend :
```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

Frontend :
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

## Déploiement
Backend  ’ Render  (root directory: backend)
Frontend ’ Vercel  (root directory: frontend)
Variable ’ NEXT_PUBLIC_API_URL=https://ink-detection-api.onrender.com