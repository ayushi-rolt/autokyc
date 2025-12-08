"use client"

import { useState, useRef, useEffect } from "react" // Ensure useEffect is imported
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Video, CheckCircle, AlertCircle, Camera, ArrowLeft, ArrowRight } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

export default function VideoCaptureePage() {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [cameraPermission, setCameraPermission] = useState<"granted" | "denied" | "prompt">("prompt")
  const [livenessDetected, setLivenessDetected] = useState(false)
  const [selfieTaken, setSelfieTaken] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const currentStep = 1

  // --- 1. THIS IS THE FIX ---
  // Use a useEffect to safely attach the stream to the video element.
  // This runs after the component renders and when the 'stream' state changes.
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream
      // Explicitly call play() in case autoPlay fails
      videoRef.current.play().catch(err => {
        console.error("Video play failed:", err)
      })
    }
  }, [stream]) // Dependency: Run this effect when 'stream' changes

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false, 
      })
      // 2. We ONLY update the state here. The useEffect will handle the rest.
      setStream(mediaStream) 
      setCameraPermission("granted")
      
      // (The line 'videoRef.current.srcObject = ...' was removed from here)
      
      // Simulate liveness detection after 2 seconds
      setTimeout(() => setLivenessDetected(true), 2000)
    } catch (error) {
      setCameraPermission("denied")
      console.error("Camera access denied:", error)
    }
  }

  // --- (Rest of the file is unchanged) ---

  const capturePhotoAndUpload = () => {
    if (!videoRef.current) return

    setIsProcessing(true)
    setError(null)

    const canvas = document.createElement("canvas")
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const context = canvas.getContext("2d")
    context?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError("Failed to capture image.")
        setIsProcessing(false)
        return
      }

      const formData = new FormData()
      formData.append("file", blob, "selfie.jpg") 

      try {
        const response = await fetch("http://127.0.0.1:8000/capture-selfie/", {
          method: "POST",
          body: formData,
        })

        const data = await response.json()

        if (!response.ok) {
          throw new Error(data.error || "Failed to upload selfie.")
        }

        setSelfieTaken(true)
        
        stream?.getTracks().forEach(track => track.stop());
        setStream(null)

      } catch (err: any) {
        setError(err.message)
      } finally {
        setIsProcessing(false)
      }
    }, "image/jpeg", 0.9)
  }

  const proceedToDocuments = () => {
    router.push("/kyc/document-upload")
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link
            href="/kyc/start"
            className="flex items-center space-x-2 text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>Back</span>
          </Link>
          <ThemeToggle />
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Step {currentStep} of 4</span>
            <span className="text-sm text-gray-500 dark:text-gray-400">Video Verification</span>
          </div>
          <Progress value={25} className="h-2" />
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Video Capture Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Video className="h-5 w-5" />
                <span>Live Selfie Capture</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
                {cameraPermission === "granted" ? (
                  <>
                    {/* Make sure these props are here: autoPlay, playsInline, muted */}
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      muted 
                      className="w-full h-full object-cover" 
                    />
                    <div className="absolute top-4 right-4">
                      {livenessDetected ? (
                        <Badge variant="default" className="bg-green-500">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Liveness Detected
                        </Badge>
                      ) : (
                        <Badge variant="secondary">
                          <AlertCircle className="h-3 w-3 mr-1" />
                          Detecting...
                        </Badge>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-white">
                    <Camera className="h-16 w-16 mb-4 opacity-50" />
                    <p className="text-center mb-4">
                      {cameraPermission === "denied"
                        ? "Camera access denied. Please enable camera permissions."
                        : "Click to enable camera access"}
                    </p>
                  </div>
                )}
                
                {selfieTaken && (
                    <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
                        <CheckCircle className="h-16 w-16 text-green-500 mb-4" />
                        <p className="text-white text-xl font-medium">Selfie Captured!</p>
                    </div>
                )}
              </div>

              <div className="flex justify-center space-x-4">
                {cameraPermission !== "granted" ? (
                  <Button onClick={startCamera} size="lg">
                    <Camera className="h-4 w-4 mr-2" />
                    Enable Camera
                  </Button>
                ) : (
                  !selfieTaken && (
                    <Button
                      onClick={capturePhotoAndUpload}
                      size="lg"
                      disabled={!livenessDetected || isProcessing}
                      className="bg-red-600 hover:bg-red-700"
                    >
                      <Camera className="h-4 w-4 mr-2" />
                      {isProcessing ? "Processing..." : "Capture Selfie"}
                    </Button>
                  )
                )}
              </div>

              {error && (
                <div className="flex items-center space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="h-5 w-5 text-red-600" />
                  <span className="text-red-800 text-sm">{error}</span>
                </div>
              )}

              {selfieTaken && (
                <div className="text-center">
                  <Button onClick={proceedToDocuments} size="lg" className="w-full">
                    Continue to Document Upload
                    <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Instructions Section */}
          <Card>
            <CardHeader>
              <CardTitle>Capture Instructions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">1</div>
                  <div>
                    <p className="font-medium">Position yourself properly</p>
                    <p className="text-sm text-gray-600">Ensure your face is clearly visible and well-lit</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">2</div>
                  <div>
                    <p className="font-medium">Wait for liveness detection</p>
                    <p className="text-sm text-gray-600">The system will verify you are a real person</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">3</div>
                  <div>
                    <p className="font-medium">Capture your selfie</p>
                    <p className="text-sm text-gray-600">Click the "Capture Selfie" button when ready. Hold still.</p>
                  </div>
                </div>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">Tips for best results:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Use good lighting (avoid backlighting)</li>
                  <li>• Look directly at the camera</li>
                  <li>• Hold your device steady</li>
                  <li>• Ensure stable internet connection</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}