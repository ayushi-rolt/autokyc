"use client"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Video, Play, Square, CheckCircle, AlertCircle, Camera, ArrowLeft, ArrowRight } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

export default function VideoCaptureePage() {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [cameraPermission, setCameraPermission] = useState<"granted" | "denied" | "prompt">("prompt")
  const [livenessDetected, setLivenessDetected] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [currentStep, setCurrentStep] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isRecording])

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      })
      setStream(mediaStream)
      setCameraPermission("granted")
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setTimeout(() => setLivenessDetected(true), 2000)
    } catch (error) {
      setCameraPermission("denied")
      console.error("Camera access denied:", error)
    }
  }

  const startRecording = () => {
    if (stream) {
      const mediaRecorder = new MediaRecorder(stream)
      const chunks: BlobPart[] = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" })
        setRecordedBlob(blob)
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const proceedToDocuments = async () => {
    if (!recordedBlob) return

    setLoading(true)
    setError(null)

    try {
      const userId = sessionStorage.getItem("userId")
      if (!userId) {
        throw new Error("User ID not found. Please start from the beginning.")
      }

      const formData = new FormData()
      formData.append("video", recordedBlob, "verification_video.webm")
      formData.append("user_id", userId)

      const response = await fetch("/api/kyc/submit-video", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to upload video")
      }

      router.push("/kyc/document-upload")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      console.error("Error uploading video:", err)
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
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
                <span>Video Capture</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm">{error}</div>
              )}

              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
                {cameraPermission === "granted" ? (
                  <>
                    <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
                    {isRecording && (
                      <div className="absolute top-4 left-4 flex items-center space-x-2">
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-white text-sm font-medium">REC {formatTime(recordingTime)}</span>
                      </div>
                    )}
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
              </div>

              <div className="flex justify-center space-x-4">
                {cameraPermission !== "granted" ? (
                  <Button onClick={startCamera} size="lg">
                    <Camera className="h-4 w-4 mr-2" />
                    Enable Camera
                  </Button>
                ) : (
                  <>
                    {!isRecording ? (
                      <Button
                        onClick={startRecording}
                        size="lg"
                        disabled={!livenessDetected}
                        className="bg-red-600 hover:bg-red-700"
                      >
                        <Play className="h-4 w-4 mr-2" />
                        Start Recording
                      </Button>
                    ) : (
                      <Button onClick={stopRecording} size="lg" variant="outline">
                        <Square className="h-4 w-4 mr-2" />
                        Stop Recording
                      </Button>
                    )}
                  </>
                )}
              </div>

              {recordedBlob && (
                <div className="text-center">
                  <Badge variant="default" className="bg-green-500 mb-4">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Video Recorded Successfully
                  </Badge>
                  <Button onClick={proceedToDocuments} size="lg" className="w-full" disabled={loading}>
                    {loading ? "Uploading..." : "Continue to Document Upload"}
                    <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Instructions Section */}
          <Card>
            <CardHeader>
              <CardTitle>Recording Instructions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                    1
                  </div>
                  <div>
                    <p className="font-medium">Position yourself properly</p>
                    <p className="text-sm text-gray-600">Ensure your face is clearly visible and well-lit</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                    2
                  </div>
                  <div>
                    <p className="font-medium">Wait for liveness detection</p>
                    <p className="text-sm text-gray-600">The system will verify you are a real person</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                    3
                  </div>
                  <div>
                    <p className="font-medium">Record a 10-15 second video</p>
                    <p className="text-sm text-gray-600">Speak clearly and look directly at the camera</p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                    4
                  </div>
                  <div>
                    <p className="font-medium">Say the following phrase</p>
                    <p className="text-sm text-gray-600 font-mono bg-gray-100 p-2 rounded">
                      "I am [Your Full Name] and I am completing my KYC verification today"
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">Tips for best results:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Use good lighting (avoid backlighting)</li>
                  <li>• Keep your device steady</li>
                  <li>• Speak clearly and at normal pace</li>
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
