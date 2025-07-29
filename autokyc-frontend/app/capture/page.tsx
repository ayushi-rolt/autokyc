"use client"

import { useState, useRef, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Camera, RotateCcw, Check, AlertCircle, User } from "lucide-react"
import { useRouter } from "next/navigation"
import { apiClient } from "@/app/api/config"

export default function CapturePage() {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState("")
  const [cameraActive, setCameraActive] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const router = useRouter()

  const startCamera = useCallback(async () => {
    try {
      setError("")
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      })

      setStream(mediaStream)
      setCameraActive(true)

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (err) {
      setError("Unable to access camera. Please ensure camera permissions are granted.")
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
      setCameraActive(false)
    }
  }, [stream])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext("2d")

    if (!context) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0)

    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.8)
    setCapturedImage(imageDataUrl)
    stopCamera()
  }, [stopCamera])

  const retakePhoto = useCallback(() => {
    setCapturedImage(null)
    startCamera()
  }, [startCamera])

  const dataURLtoBlob = (dataURL: string): Blob => {
    const arr = dataURL.split(",")
    const mime = arr[0].match(/:(.*?);/)?.[1] || "image/jpeg"
    const bstr = atob(arr[1])
    let n = bstr.length
    const u8arr = new Uint8Array(n)
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n)
    }
    return new Blob([u8arr], { type: mime })
  }

  const processSelfie = async () => {
    if (!capturedImage) return

    setIsProcessing(true)
    setError("")

    try {
      // Convert captured image to blob
      const blob = dataURLtoBlob(capturedImage)
      const formData = new FormData()
      formData.append("file", blob, "selfie.jpg")

      // Call capture-selfie endpoint
      const captureResult = await apiClient.captureSelfie(formData)

      // Get embedding from selfie
      const embeddingResult = await apiClient.getEmbeddingFromSelfie()

      // Store results and redirect to verification
      sessionStorage.setItem(
        "selfieData",
        JSON.stringify({
          captureResult,
          embeddingResult,
          imageUrl: capturedImage,
        }),
      )

      router.push("/verify")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process selfie")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Capture Selfie</h1>
          <p className="text-gray-600">Take a clear photo of your face for verification</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="w-5 h-5" />
              <span>Face Capture</span>
            </CardTitle>
            <CardDescription>Position your face in the center of the frame and ensure good lighting</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Camera/Image Display */}
            <div className="relative bg-gray-100 rounded-lg overflow-hidden aspect-[4/3]">
              {!cameraActive && !capturedImage && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-500">Click "Start Camera" to begin</p>
                  </div>
                </div>
              )}

              {cameraActive && (
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              )}

              {capturedImage && (
                <img
                  src={capturedImage || "/placeholder.svg"}
                  alt="Captured selfie"
                  className="w-full h-full object-cover"
                />
              )}

              {/* Face guide overlay */}
              {cameraActive && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-64 h-80 border-2 border-white rounded-full opacity-50"></div>
                </div>
              )}
            </div>

            <canvas ref={canvasRef} className="hidden" />

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Camera Controls */}
            <div className="flex justify-center space-x-4">
              {!cameraActive && !capturedImage && (
                <Button onClick={startCamera} size="lg">
                  <Camera className="w-4 h-4 mr-2" />
                  Start Camera
                </Button>
              )}

              {cameraActive && (
                <>
                  <Button onClick={capturePhoto} size="lg">
                    <Camera className="w-4 h-4 mr-2" />
                    Capture Photo
                  </Button>
                  <Button onClick={stopCamera} variant="outline" size="lg">
                    Cancel
                  </Button>
                </>
              )}

              {capturedImage && (
                <>
                  <Button onClick={retakePhoto} variant="outline" size="lg">
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Retake
                  </Button>
                  <Button onClick={processSelfie} size="lg" disabled={isProcessing}>
                    <Check className="w-4 h-4 mr-2" />
                    {isProcessing ? "Processing..." : "Continue"}
                  </Button>
                </>
              )}
            </div>

            {/* Instructions */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-medium text-blue-900 mb-2">Photo Guidelines:</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Look directly at the camera</li>
                <li>• Ensure your face is well-lit</li>
                <li>• Remove sunglasses or hats</li>
                <li>• Keep a neutral expression</li>
                <li>• Make sure your entire face is visible</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
