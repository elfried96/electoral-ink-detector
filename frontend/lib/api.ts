const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface AnalysisResult {
  success: boolean;
  ink_detected: boolean;
  voted: boolean;
  verdict: "CERTAIN" | "ABSENT" | "ERREUR";
  score_global: number;
  doigt_encre: 'pouce' | 'index' | null;
  fraud: {
    suspected: boolean;
    score: number;
    indicators: string[];
  };
  doigts: {
    pouce?: {
      ink_detected: boolean;
      score_pct: number;
      confidence: number;
    };
    index?: {
      ink_detected: boolean;
      score_pct: number;
      confidence: number;
    };
  };
  processing_time_ms: number;
  error?: string;
  error_code?: string;
}

// Mapper les codes d'erreur vers des messages clairs
const ERROR_MESSAGES: Record<string, { message: string; icon: string }> = {
  IMAGE_TOO_SMALL: {
    message: "Photo trop petite. Approchez-vous.",
    icon: "🔍"
  },
  IMAGE_BLURRY: {
    message: "Photo floue. Tenez le téléphone stable.",
    icon: "📸"
  },
  BAD_LIGHTING: {
    message: "Éclairage insuffisant. Allez vers une fenêtre.",
    icon: "💡"
  },
  NO_HAND_DETECTED: {
    message: "Main non détectée. Montrez paume ouverte.",
    icon: "✋"
  },
  TOO_MANY_HANDS: {
    message: "Une seule main à la fois.",
    icon: "🤚"
  },
  TIMEOUT: {
    message: "Serveur occupé. Réessayez dans 30 secondes.",
    icon: "⏳"
  },
  TOO_MANY_REQUESTS: {
    message: "Trop de tentatives. Attendez 1 minute.",
    icon: "⏸️"
  },
  SERVER_STARTING: {
    message: "Serveur en cours de démarrage, patience...",
    icon: "🔄"
  }
};

export function getErrorMessage(code?: string): { message: string; icon: string } {
  return ERROR_MESSAGES[code || ''] || {
    message: 'Erreur inconnue. Veuillez réessayer.',
    icon: '❌'
  };
}

// Fonction avec timeout personnalisé
async function fetchWithTimeout(url: string, options: RequestInit, timeout = 60000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('TIMEOUT');
    }
    throw error;
  }
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 60000) // 60s

  try {
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    })

    clearTimeout(timeout)

    if (!res.ok) {
      const err = await res.json()
      throw { code: err.error, message: err.message }
    }

    return await res.json()

  } catch (e: any) {
    clearTimeout(timeout)
    if (e.name === 'AbortError') {
      throw { code: 'NETWORK_ERROR', message: 'Délai dépassé. Le serveur met du temps à répondre.' }
    }
    throw e
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(`${API_URL}/health`, {}, 3000); // 3 seconds timeout
    const data = await response.json();
    return data.status === 'ok' && data.model_loaded === true;
  } catch (error) {
    return false;
  }
}

// Warm-up au démarrage de l'app (évite le cold start)
export async function warmupServer(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) })
    return res.ok
  } catch {
    return false
  }
}