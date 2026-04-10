'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, Upload, X } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  onClose: () => void;
}

export default function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [isCapturing, setIsCapturing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mode, setMode] = useState<'camera' | 'upload'>('camera');

  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);
      
      // Vérifier d'abord si l'API est disponible
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not available');
      }
      
      // Essayer d'abord avec facingMode environment (caméra arrière)
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'environment',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });
      } catch (envErr) {
        // Si ça échoue, essayer avec n'importe quelle caméra
        stream = await navigator.mediaDevices.getUserMedia({
          video: true
        });
      }
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (err: any) {
      console.error('Erreur caméra:', err);
      let errorMessage = 'Impossible d\'accéder à la caméra.';
      
      if (err.name === 'NotAllowedError') {
        errorMessage = 'Permission caméra refusée. Veuillez autoriser l\'accès dans les paramètres.';
      } else if (err.name === 'NotFoundError') {
        errorMessage = 'Aucune caméra détectée sur cet appareil.';
      } else if (err.name === 'NotReadableError') {
        errorMessage = 'Caméra déjà utilisée par une autre application.';
      } else if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
        errorMessage = 'La caméra nécessite une connexion sécurisée (HTTPS).';
      }
      
      setCameraError(errorMessage);
      // Basculer automatiquement vers l'upload si la caméra échoue
      setTimeout(() => setMode('upload'), 2000);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
    
    return () => {
      stopCamera();
    };
  }, [mode, startCamera, stopCamera]);

  // Fonction améliorée pour compresser l'image
  async function compressImage(file: File): Promise<File> {
    return new Promise((resolve) => {
      const img = new Image()
      const url = URL.createObjectURL(file)
      img.onload = () => {
        const MAX = 800
        let { width, height } = img
        if (width > MAX || height > MAX) {
          if (width > height) { 
            height = Math.round(height * MAX / width); 
            width = MAX 
          } else { 
            width = Math.round(width * MAX / height); 
            height = MAX 
          }
        }
        const canvas = document.createElement('canvas')
        canvas.width = width; 
        canvas.height = height
        canvas.getContext('2d')!.drawImage(img, 0, 0, width, height)
        canvas.toBlob(
          (blob) => {
            const compressedFile = new File([blob!], file.name || 'capture.jpg', { type: 'image/jpeg' })
            console.log(`Image compressée: ${(compressedFile.size / 1024).toFixed(0)} KB (${width}x${height})`)
            resolve(compressedFile)
          },
          'image/jpeg', 
          0.82
        )
        URL.revokeObjectURL(url)
      }
      img.src = url
    })
  }

  // Fonction pour redimensionner et compresser l'image
  const processImage = async (imageSource: HTMLVideoElement | HTMLImageElement | File): Promise<File> => {
    const canvas = canvasRef.current;
    if (!canvas) throw new Error('Canvas not available');

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas context not available');

    let img: HTMLImageElement | HTMLVideoElement;
    
    if (imageSource instanceof File) {
      // Si c'est un fichier, utiliser la fonction de compression
      return compressImage(imageSource);
    } else {
      img = imageSource;
    }

    // Calculer les dimensions (max 800x800)
    const maxSize = 800;
    let width = img instanceof HTMLVideoElement ? img.videoWidth : img.width;
    let height = img instanceof HTMLVideoElement ? img.videoHeight : img.height;
    
    if (width > maxSize || height > maxSize) {
      const scale = Math.min(maxSize / width, maxSize / height);
      width = Math.round(width * scale);
      height = Math.round(height * scale);
    }

    // Redimensionner sur le canvas
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    // Compresser en JPEG avec qualité 0.82
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            console.log(`Image: ${(file.size / 1024).toFixed(0)} KB (${width}x${height})`);
            resolve(file);
          } else {
            reject(new Error('Failed to create blob'));
          }
        },
        'image/jpeg',
        0.82  // Qualité optimale
      );
    });
  };

  const capturePhoto = async () => {
    if (videoRef.current && canvasRef.current) {
      try {
        const file = await processImage(videoRef.current);
        onCapture(file);
        stopCamera();
      } catch (error) {
        console.error('Erreur lors de la capture:', error);
      }
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        const processedFile = await processImage(file);
        onCapture(processedFile);
        stopCamera();
      } catch (error) {
        console.error('Erreur lors du traitement de l\'image:', error);
      }
    }
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="relative flex-1">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-20 bg-black/50 text-white rounded-full p-2"
        >
          <X size={24} />
        </button>

        {mode === 'camera' ? (
          <>
            {cameraError ? (
              <div className="flex items-center justify-center h-full text-white p-4">
                <div className="text-center">
                  <p className="mb-4">{cameraError}</p>
                  <button
                    onClick={() => setMode('upload')}
                    className="bg-blue-500 text-white px-4 py-2 rounded-lg"
                  >
                    Utiliser la galerie
                  </button>
                </div>
              </div>
            ) : (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full object-cover"
                />
                
                <div style={{position:'absolute', inset:0, pointerEvents:'none'}}>
                  {/* Silhouette main centrée */}
                  <div style={{
                    position:'absolute', top:'15%', left:'50%',
                    transform:'translateX(-50%)',
                    width:120, height:160,
                    border:'2px solid rgba(255,255,255,0.6)',
                    borderRadius:'60px 60px 40px 40px',
                    boxShadow:'0 0 0 9999px rgba(0,0,0,0.35)'
                  }}/>
                  {/* Texte guide */}
                  <div style={{
                    position:'absolute', bottom:24, left:0, right:0,
                    textAlign:'center', color:'white',
                    fontSize:14, fontWeight:500,
                    textShadow:'0 1px 4px rgba(0,0,0,0.8)',
                    padding:'0 16px'
                  }}>
                    Centrez votre main — pouce et index bien visibles
                  </div>
                </div>
              </>
            )}
          </>
        ) : (
          <div className="flex items-center justify-center h-full bg-gray-900">
            <div className="text-center text-white p-8">
              <Upload size={64} className="mx-auto mb-4 text-gray-400" />
              <p className="mb-4 text-lg">Sélectionnez une image depuis votre galerie</p>
              <label className="bg-blue-500 text-white px-6 py-3 rounded-lg inline-block cursor-pointer">
                Choisir une image
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>
            </div>
          </div>
        )}
        
        <canvas ref={canvasRef} className="hidden" />
      </div>
      
      <div className="bg-black p-4 flex gap-2">
        {mode === 'camera' ? (
          <>
            <button
              onClick={capturePhoto}
              disabled={!!cameraError}
              className="flex-1 bg-white text-black py-4 rounded-full font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Camera size={24} />
              Prendre une photo
            </button>
            <button
              onClick={() => setMode('upload')}
              className="bg-gray-700 text-white px-6 py-4 rounded-full"
            >
              <Upload size={24} />
            </button>
          </>
        ) : (
          <button
            onClick={() => setMode('camera')}
            className="flex-1 bg-gray-700 text-white py-4 rounded-full font-semibold text-lg flex items-center justify-center gap-2"
          >
            <Camera size={24} />
            Utiliser la caméra
          </button>
        )}
      </div>
    </div>
  );
}