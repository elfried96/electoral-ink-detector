'use client';

import React from 'react';
import { AnalysisResult } from '@/lib/api';
import { CheckCircle, XCircle, AlertCircle, Clock, AlertTriangle, RefreshCw, Fingerprint } from 'lucide-react';

interface ResultCardProps {
  result: AnalysisResult;
  onRetry: () => void;
}

export default function ResultCard({ result, onRetry }: ResultCardProps) {
  // Déterminer l'état principal selon la priorité définie
  const getMainState = () => {
    if (result.fraud?.suspected) {
      return {
        type: 'fraud',
        bgColor: 'from-red-500 to-red-600',
        icon: <AlertTriangle size={48} className="text-white" />,
        label: 'Attention requise',
        title: 'Encre détectée — image suspecte',
        subtitle: 'Des signes de retouche ont été détectés. Vérification manuelle requise.',
      };
    }
    
    if (result.voted && result.verdict === 'CERTAIN') {
      const doigtText = result.doigt_encre === 'pouce' ? 'pouce' : 'index';
      return {
        type: 'voted_certain',
        bgColor: 'from-green-500 to-green-600',
        icon: <CheckCircle size={48} className="text-white" />,
        label: 'Vérification terminée',
        title: 'Cette personne a voté',
        subtitle: `Encre détectée sur le ${doigtText}.`,
      };
    }
    
    return {
      type: 'not_voted',
      bgColor: 'from-red-500 to-red-600',
      icon: <XCircle size={48} className="text-white" />,
      label: 'Vérification terminée',
      title: 'Aucune encre détectée',
      subtitle: "Aucun signe d'encre électorale sur les doigts analysés.",
    };
  };

  const state = getMainState();

  // Déterminer le niveau de risque de fraude
  const getFraudLevel = () => {
    const score = result.fraud?.score || 0;
    if (score < 30) {
      return {
        level: 'Faible',
        color: 'bg-green-500 text-white',
        message: 'Aucun signe de manipulation détecté.',
      };
    }
    if (score < 50) {
      return {
        level: 'Moyen',
        color: 'bg-orange-500 text-white',
        message: `Quelques anomalies mineures. Score : ${score}/100`,
      };
    }
    return {
      level: 'Élevé',
      color: 'bg-red-500 text-white',
      message: `Score de risque : ${score}/100. Vérification visuelle recommandée.`,
    };
  };

  const fraudLevel = getFraudLevel();
  const showFraudIndicators = (result.fraud?.score || 0) >= 30;

  // Label pour la barre de confiance
  const confidenceLabel = state.type === 'fraud' ? "Score d'encre détectée" : 'Niveau de certitude';

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-md mx-auto px-4 py-6">
        
        {/* 1. ZONE VERDICT */}
        <div className={`bg-gradient-to-br ${state.bgColor} rounded-2xl p-6 mb-4 text-white shadow-xl`}>
          <div className="flex flex-col items-center text-center">
            <div className="w-20 h-20 bg-white/20 backdrop-blur rounded-full flex items-center justify-center mb-3">
              {state.icon}
            </div>
            <span className="text-sm bg-white/20 px-3 py-1 rounded-full mb-3">
              {state.label}
            </span>
            <h1 className="text-2xl font-bold mb-2">
              {state.title}
            </h1>
            <p className="text-white/90">
              {state.subtitle}
            </p>
          </div>
        </div>

        {/* 2. BANNIÈRE FRAUDE */}
        {result.fraud?.suspected && (
          <div className="bg-orange-100 border-l-4 border-orange-500 p-4 rounded-lg mb-4">
            <p className="text-orange-800 font-semibold">
              Vérification manuelle recommandée
            </p>
          </div>
        )}

        {/* 3. BARRE DE CONFIANCE */}
        <div className="bg-white rounded-xl p-5 mb-4 shadow-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-700 font-medium">{confidenceLabel}</span>
            <span className="text-2xl font-bold text-gray-900">
              {Math.round(result.score_global)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-700 ${
                result.score_global >= 7 ? 'bg-green-500' :
                result.score_global >= 3 ? 'bg-orange-500' : 'bg-gray-400'
              }`}
              style={{ width: `${Math.min(result.score_global * 10, 100)}%` }}
            />
          </div>
        </div>

        {/* 4. DÉTAIL PAR DOIGT */}
        <div className="bg-white rounded-xl p-5 mb-4 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">Détail par doigt</h3>
          
          <div className="space-y-3">
            {/* Pouce */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-lg">👍</span>
                </div>
                <span className="font-medium text-gray-700">
                  Pouce
                  {result.doigt_encre === 'pouce' && (
                    <span className="ml-2 text-green-600 text-sm">✓ Vote confirmé</span>
                  )}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-medium">
                  {Math.round(result.doigts?.pouce?.score_pct || 0)}%
                </span>
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  result.doigts?.pouce?.ink_detected 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {result.doigts?.pouce?.ink_detected ? 'ENCRE DÉTECTÉE' : 'Propre'}
                </span>
              </div>
            </div>

            {/* Index */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-lg">👆</span>
                </div>
                <span className="font-medium text-gray-700">
                  Index
                  {result.doigt_encre === 'index' && (
                    <span className="ml-2 text-green-600 text-sm">✓ Vote confirmé</span>
                  )}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-medium">
                  {Math.round(result.doigts?.index?.score_pct || 0)}%
                </span>
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  result.doigts?.index?.ink_detected 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {result.doigts?.index?.ink_detected ? 'ENCRE DÉTECTÉE' : 'Propre'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* 5. RISQUE DE FRAUDE */}
        <div className="bg-gray-100 rounded-xl p-5 mb-4">
          <h3 className="font-semibold text-gray-800 mb-3">Risque de fraude</h3>
          
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-600">Niveau de risque</span>
            <span className={`px-3 py-1 rounded-full text-sm font-bold ${fraudLevel.color}`}>
              {fraudLevel.level}
            </span>
          </div>
          
          <p className="text-gray-700 text-sm mb-3">
            {fraudLevel.message}
          </p>
          
          {showFraudIndicators && result.fraud?.indicators && result.fraud.indicators.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <p className="text-sm text-gray-600 mb-2">Indicateurs détectés :</p>
              <ul className="space-y-1">
                {result.fraud.indicators.map((indicator, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>{indicator}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* 6. MÉTA-INFORMATIONS */}
        <div className="flex gap-3 mb-4">
          {result.doigt_encre && (
            <span className="bg-green-100 px-4 py-2 rounded-full text-sm text-green-700">
              <Fingerprint size={16} className="inline mr-1" />
              Encre sur : {result.doigt_encre === 'pouce' ? 'Pouce' : 'Index'}
            </span>
          )}
          <span className="bg-gray-100 px-4 py-2 rounded-full text-sm text-gray-700">
            <Clock size={16} className="inline mr-1" />
            {result.processing_time_ms}ms
          </span>
        </div>

        {/* 7. RAPPORT PRÉTRAITEMENT */}
        {result.preprocessing && (
          <div className="bg-gray-50 rounded-lg px-4 py-3 mb-6 border border-gray-200">
            <div className="flex items-center gap-2 text-xs text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              {result.preprocessing.n_corrections > 0 ? (
                <span>
                  Photo corrigée automatiquement : {result.preprocessing.corrections_appliquees.join(' · ')}
                </span>
              ) : (
                <span>Photo de bonne qualité</span>
              )}
            </div>
            {result.preprocessing.n_corrections > 0 && (
              <div className="text-xs text-gray-500 mt-1 ml-6">
                {result.preprocessing.taille_originale} → {result.preprocessing.taille_finale}
              </div>
            )}
          </div>
        )}

        {/* 8. BOUTON */}
        <button
          onClick={onRetry}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white py-4 rounded-xl font-semibold text-lg transition-colors flex items-center justify-center gap-2 min-h-[56px] shadow-lg"
        >
          <RefreshCw size={20} />
          Analyser une autre photo
        </button>
      </div>
    </div>
  );
}