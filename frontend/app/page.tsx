'use client';

import { useState, useEffect, useCallback } from 'react';
import CameraCapture from '@/components/CameraCapture';
import ResultCard from '@/components/ResultCard';
import ErrorCard from '@/components/ErrorCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import SimpleUpload from '@/components/SimpleUpload';
import { analyzeImage, checkHealth, warmupServer, AnalysisResult, getErrorMessage } from '@/lib/api';
import { Camera, Upload, Fingerprint, Shield, ChevronRight } from 'lucide-react';

type AppState = 'home' | 'camera' | 'preview' | 'loading' | 'result' | 'error';

export default function Home() {
  const [appState, setAppState] = useState<AppState>('home');
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const [serverWarning, setServerWarning] = useState(false);
  const [analysisTimeout, setAnalysisTimeout] = useState<NodeJS.Timeout | null>(null);
  const [errorData, setErrorData] = useState<{ code: string; message: string } | null>(null);

  useEffect(() => {
    // Warmup initial
    warmupServer().then(ok => {
      if (!ok) {
        // Afficher banner discret "Démarrage du serveur (~30s)..."
        setServerWarning(true);
        // Réessayer toutes les 5s pendant max 60s
        const interval = setInterval(async () => {
          const ready = await warmupServer();
          if (ready) { 
            setServerWarning(false); 
            clearInterval(interval); 
          }
        }, 5000);
        setTimeout(() => clearInterval(interval), 60000);
      }
    });
    
    // Keep-alive : ping toutes les 5 minutes pour éviter le sleep
    const keepAlive = setInterval(() => {
      warmupServer();
    }, 5 * 60 * 1000); // 5 minutes
    
    return () => clearInterval(keepAlive);
  }, []);

  const handleCapture = useCallback((file: File) => {
    setCapturedImage(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setAppState('preview');
  }, []);

  const handleAnalyze = async () => {
    if (!capturedImage) return;
    
    setAppState('loading');
    setError(null);
    
    try {
      // Afficher message d'attente après 10 secondes
      const timeoutId = setTimeout(() => {
        setError('SERVER_STARTING');
      }, 10000);
      setAnalysisTimeout(timeoutId);
      
      const result = await analyzeImage(capturedImage);
      clearTimeout(timeoutId);
      setAnalysisTimeout(null);
      setAnalysisResult(result);
      setAppState('result');
    } catch (err: any) {
      if (analysisTimeout) {
        clearTimeout(analysisTimeout);
        setAnalysisTimeout(null);
      }
      setErrorData(err);
      setError(err.code || err.message || 'Erreur lors de l\'analyse');
      // Si c'est une erreur spécifique, afficher ErrorCard
      if (err.code && ['IMAGE_TOO_SMALL', 'IMAGE_BLURRY', 'BAD_LIGHTING', 'NO_HAND_DETECTED', 'TOO_MANY_REQUESTS', 'NETWORK_ERROR'].includes(err.code)) {
        setAppState('error');
      } else {
        setAppState('preview');
      }
    }
  };

  const handleRetry = () => {
    setCapturedImage(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setError(null);
    setAppState('home');
  };

  const handleReCapture = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setCapturedImage(null);
    setPreviewUrl(null);
    setError(null);
    setAppState('camera');
  };

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  if (appState === 'loading') {
    return <LoadingSpinner />;
  }

  if (appState === 'error' && errorData) {
    return <ErrorCard error={errorData.code} message={errorData.message} onRetry={handleRetry} />;
  }

  if (appState === 'result' && analysisResult) {
    return <ResultCard result={analysisResult} onRetry={handleRetry} />;
  }

  if (appState === 'camera') {
    return (
      <CameraCapture 
        onCapture={handleCapture} 
        onClose={() => setAppState('home')}
      />
    );
  }

  if (appState === 'preview' && previewUrl) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="max-w-md mx-auto px-4 py-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Prévisualisation</h2>
            <p className="text-gray-600 mt-1">Vérifiez que les doigts sont bien visibles</p>
          </div>
          
          <div className="rounded-2xl overflow-hidden shadow-xl bg-white">
            <img 
              src={previewUrl} 
              alt="Capture" 
              className="w-full h-auto"
            />
          </div>

          {error && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
              <div className="flex items-start gap-3">
                <span className="text-2xl">{getErrorMessage(error).icon}</span>
                <div className="flex-1">
                  <p className="text-red-700 font-medium">{getErrorMessage(error).message}</p>
                </div>
              </div>
            </div>
          )}

          <div className="mt-6 space-y-3">
            <button
              onClick={handleAnalyze}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 px-6 rounded-2xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2"
            >
              <Shield className="w-5 h-5" />
              Analyser l'image
            </button>
            
            <button
              onClick={handleReCapture}
              className="w-full bg-white border-2 border-gray-300 text-gray-700 py-4 px-6 rounded-2xl font-semibold text-lg hover:bg-gray-50 transition-colors"
            >
              Reprendre la photo
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800">
      {/* Hero Section */}
      <div className="px-4 pt-12 pb-8">
        <div className="text-center text-white">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-white/20 backdrop-blur-lg rounded-3xl mb-6 shadow-2xl">
            <Fingerprint size={48} className="text-white" />
          </div>
          <h1 className="text-3xl font-bold mb-2">
            Vérification Électorale
          </h1>
          <p className="text-blue-100 text-lg">
            Détection instantanée d'encre électorale
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 bg-white rounded-t-[40px] shadow-2xl">
        <div className="max-w-md mx-auto px-6 py-8">
          {serverWarning && (
            <div className="mb-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded-lg">
              <div className="flex items-start gap-3">
                <span className="text-2xl animate-spin">🔄</span>
                <div>
                  <p className="text-amber-800 font-semibold">
                    Démarrage du serveur
                  </p>
                  <p className="text-amber-700 text-sm mt-1">
                    Démarrage en cours (~30s)...
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Action Cards */}
          <div className="space-y-4">
            <button
              onClick={() => setAppState('camera')}
              className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-2xl p-6 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center">
                    <Camera size={28} />
                  </div>
                  <div className="text-left">
                    <p className="font-bold text-xl">Prendre une photo</p>
                    <p className="text-blue-100 text-sm">Utilisez la caméra</p>
                  </div>
                </div>
                <ChevronRight size={24} className="text-blue-200" />
              </div>
            </button>
            
            <div className="w-full bg-white border-2 border-gray-200 rounded-2xl p-6 shadow-md hover:shadow-lg hover:border-gray-300 transform hover:-translate-y-0.5 transition-all duration-200">
              <SimpleUpload onCapture={handleCapture} />
              <ChevronRight size={24} className="text-gray-400 float-right -mt-10" />
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-10 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6">
            <h3 className="font-bold text-gray-800 mb-4">Comment procéder ?</h3>
            <div className="space-y-3">
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">
                  1
                </div>
                <p className="text-gray-700">Montrez clairement le pouce et l'index</p>
              </div>
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">
                  2
                </div>
                <p className="text-gray-700">Assurez-vous d'un bon éclairage</p>
              </div>
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">
                  3
                </div>
                <p className="text-gray-700">Recevez le verdict instantanément</p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-8 text-center">
            <p className="text-xs text-gray-500">
              Technologie d'IA avancée pour une détection précise
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}