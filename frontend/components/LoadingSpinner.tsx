'use client';

import React from 'react';

export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="relative">
        <div className="w-20 h-20 border-4 border-gray-200 rounded-full animate-pulse"></div>
        <div className="absolute top-0 left-0 w-20 h-20 border-4 border-blue-500 rounded-full animate-spin border-t-transparent"></div>
      </div>
      <p className="mt-6 text-lg text-gray-700 font-medium animate-pulse">
        Analyse en cours...
      </p>
      <p className="mt-2 text-sm text-gray-500 text-center max-w-xs">
        Détection de l'encre électorale sur vos doigts
      </p>
    </div>
  );
}