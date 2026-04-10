'use client';

import React from 'react';
import { AlertTriangle, Camera, Sun, WifiOff, Clock, RefreshCw, Hand } from 'lucide-react';

interface ErrorCardProps {
  error: string;
  message?: string;
  onRetry: () => void;
}

const ERROR_MESSAGES: Record<string, { 
  title: string; 
  action: string; 
  icon: React.ReactNode 
}> = {
  IMAGE_TOO_SMALL: { 
    title: 'Photo trop petite', 
    action: 'Approchez-vous de la caméra', 
    icon: <Camera size={48} className="text-orange-500" />
  },
  IMAGE_BLURRY: { 
    title: 'Photo floue', 
    action: 'Tenez le téléphone bien stable', 
    icon: <Camera size={48} className="text-orange-500" />
  },
  BAD_LIGHTING: { 
    title: 'Mauvais éclairage', 
    action: 'Allez vers une fenêtre ou activez le flash', 
    icon: <Sun size={48} className="text-orange-500" />
  },
  NO_HAND_DETECTED: { 
    title: 'Main non détectée', 
    action: 'Montrez votre paume ouverte, pouce et index visibles', 
    icon: <Hand size={48} className="text-orange-500" />
  },
  TOO_MANY_REQUESTS: { 
    title: 'Trop de tentatives', 
    action: 'Attendez 1 minute avant de réessayer', 
    icon: <Clock size={48} className="text-red-500" />
  },
  NETWORK_ERROR: { 
    title: 'Serveur indisponible', 
    action: 'Vérifiez votre connexion et réessayez', 
    icon: <WifiOff size={48} className="text-red-500" />
  },
};

export default function ErrorCard({ error, message, onRetry }: ErrorCardProps) {
  const errorInfo = ERROR_MESSAGES[error] || {
    title: 'Erreur',
    action: message || 'Une erreur est survenue. Veuillez réessayer.',
    icon: <AlertTriangle size={48} className="text-red-500" />
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Icon */}
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-red-50 rounded-full flex items-center justify-center">
              {errorInfo.icon}
            </div>
          </div>

          {/* Title */}
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-3">
            {errorInfo.title}
          </h2>

          {/* Action message */}
          <p className="text-center text-gray-600 mb-8">
            {errorInfo.action}
          </p>

          {/* Additional message if provided */}
          {message && error !== message && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-6">
              <p className="text-sm text-orange-800">{message}</p>
            </div>
          )}

          {/* Retry button */}
          <button
            onClick={onRetry}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-4 rounded-xl font-semibold text-lg transition-colors flex items-center justify-center gap-2 shadow-lg"
            disabled={error === 'TOO_MANY_REQUESTS'}
          >
            <RefreshCw size={20} />
            Réessayer
          </button>

          {/* Rate limit message */}
          {error === 'TOO_MANY_REQUESTS' && (
            <p className="text-center text-sm text-gray-500 mt-3">
              Le bouton sera réactivé dans 60 secondes
            </p>
          )}
        </div>
      </div>
    </div>
  );
}