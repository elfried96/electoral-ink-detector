import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ['192.168.1.132'],
  // Pour permettre l'accès HTTPS dans un contexte non-sécurisé (localhost)
  // et gérer les permissions caméra
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Permissions-Policy',
            value: 'camera=self, microphone=self'
          }
        ]
      }
    ]
  }
};

export default nextConfig;
