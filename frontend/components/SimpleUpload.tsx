'use client';

import React, { useRef } from 'react';
import { Upload } from 'lucide-react';

interface SimpleUploadProps {
  onCapture: (file: File) => void;
  className?: string;
}

export default function SimpleUpload({ onCapture, className = "" }: SimpleUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processImage = async (file: File): Promise<File> => {
    // Créer un canvas pour redimensionner l'image
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) throw new Error('Canvas context not available');

    return new Promise((resolve, reject) => {
      img.onload = () => {
        // Calculer les nouvelles dimensions (max 800x800)
        const maxSize = 800;
        let width = img.width;
        let height = img.height;
        
        if (width > maxSize || height > maxSize) {
          const scale = Math.min(maxSize / width, maxSize / height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }

        // Redimensionner
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        // Convertir en blob JPEG
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const processedFile = new File([blob], file.name || 'image.jpg', { 
                type: 'image/jpeg' 
              });
              console.log(`Image processée: ${(processedFile.size / 1024).toFixed(0)} KB (${width}x${height})`);
              resolve(processedFile);
            } else {
              reject(new Error('Failed to create blob'));
            }
          },
          'image/jpeg',
          0.8
        );
      };

      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(file);
    });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        const processedFile = await processImage(file);
        onCapture(processedFile);
      } catch (error) {
        console.error('Erreur lors du traitement de l\'image:', error);
        // Envoyer le fichier original en cas d'erreur
        onCapture(file);
      }
    }
    // Réinitialiser l'input pour permettre de sélectionner le même fichier
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <label className={`cursor-pointer ${className}`}>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center">
            <Upload size={28} className="text-gray-600" />
          </div>
          <div className="text-left">
            <p className="font-bold text-xl text-gray-800">Importer</p>
            <p className="text-gray-500 text-sm">Depuis la galerie</p>
          </div>
        </div>
      </div>
    </label>
  );
}