// API configuration for FastAPI backend integration
export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  endpoints: {
    captureSelfie: "/capture-selfie/",
    embeddingFromSelfie: "/embedding-from-selfie/",
    verifyFace: "/verify-face/",
  },
  timeout: 30000, // 30 seconds
}

// API client utility functions
export class APIClient {
  private baseURL: string

  constructor(baseURL: string = API_CONFIG.baseURL) {
    this.baseURL = baseURL
  }

  async captureSelfie(imageData: FormData): Promise<any> {
    const response = await fetch(`${this.baseURL}${API_CONFIG.endpoints.captureSelfie}`, {
      method: "POST",
      body: imageData,
    })

    if (!response.ok) {
      throw new Error(`Selfie capture failed: ${response.statusText}`)
    }

    return response.json()
  }

  async getEmbeddingFromSelfie(): Promise<any> {
  const response = await fetch(`${this.baseURL}${API_CONFIG.endpoints.embeddingFromSelfie}`, {
    method: "GET",
  })

  if (!response.ok) {
    throw new Error(`Embedding generation failed: ${response.statusText}`)
  }

  return response.json()
}

  async verifyFace(verificationData: FormData): Promise<any> {
    const response = await fetch(`${this.baseURL}${API_CONFIG.endpoints.verifyFace}`, {
      method: "POST",
      body: verificationData,
    })

    if (!response.ok) {
      throw new Error(`Face verification failed: ${response.statusText}`)
    }

    return response.json()
  }
}

export const apiClient = new APIClient()
