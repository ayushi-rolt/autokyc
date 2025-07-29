"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Upload, CheckCircle, XCircle, AlertCircle, Camera } from "lucide-react"
import { useRouter } from "next/navigation"
import { apiClient } from "@/app/api/config"

interface SelfieData {
  captureResult: any
  embeddingResult: any
  imageUrl: string
}

export default function VerifyPage() {
  const [selfieData, setSelfieData] = useState<SelfieData | null>(null)
  const [referenceImage, setReferenceImage] = useState<File | null>(null)
  const [referencePreview, setReferencePreview] = useState<string | null>(null)
  const [isVerifying, setIsVerifying] = useState(false)
  const [verificationResult, setVerificationResult] = useState<any>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState("")

  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  useEffect(() => {
    // Get selfie data from session storage
    const storedData = sessionStorage.getItem("selfieData")
    if (storedData) {
      setSelfieData(JSON.parse(storedData))
    } else {
      // Redirect back to capture if no selfie data
      router.push("/capture")
    }
  }, [router])

  const handleReferenceUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file")
      return
    }

    setReferenceImage(file)
    setError("")

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setReferencePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const performVerification = async () => {
    if (!selfieData || !referenceImage) {
      setError("Missing selfie or reference image")
      return
    }

    setIsVerifying(true)
    setError("")
    setProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 15
        })
      }, 500)

      // Prepare form data for verification
      const formData = new FormData()

      // Convert selfie data URL back to blob
      const selfieBlob = dataURLtoBlob(selfieData.imageUrl)
      formData.append("selfie", selfieBlob, "selfie.jpg")
      formData.append("reference", referenceImage, referenceImage.name)

      // Add embedding data if available
      if (selfieData.embeddingResult) {
        formData.append("embedding_data", JSON.stringify(selfieData.embeddingResult))
      }

      // Call verify-face endpoint
      const result = await apiClient.verifyFace(formData)

      clearInterval(progressInterval)
      setProgress(100)
      setVerificationResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed")
      setProgress(0)
    } finally {
      setIsVerifying(false)
    }
  }

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

  const resetVerification = () => {
    setVerificationResult(null)
    setReferenceImage(null)
    setReferencePreview(null)
    setProgress(0)
    setError("")
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  if (!selfieData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col items-center py-8">
            <Camera className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Selfie Data</h3>
            <p className="text-gray-600 text-center mb-4">Please capture a selfie first</p>
            <Button onClick={() => router.push("/capture")}>Capture Selfie</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (verificationResult) {
    const isMatch = verificationResult.is_match || verificationResult.match || false
    const confidence = verificationResult.confidence || verificationResult.similarity_score || 0

    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-4 max-w-2xl">
          <Card>
            <CardHeader className="text-center">
              <div className="flex justify-center mb-4">
                {isMatch ? (
                  <CheckCircle className="w-16 h-16 text-green-600" />
                ) : (
                  <XCircle className="w-16 h-16 text-red-600" />
                )}
              </div>
              <CardTitle className={isMatch ? "text-green-600" : "text-red-600"}>
                {isMatch ? "Verification Successful" : "Verification Failed"}
              </CardTitle>
              <CardDescription>Face verification completed</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Results */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="text-center">
                  <div className={`text-3xl font-bold ${isMatch ? "text-green-600" : "text-red-600"}`}>
                    {(confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Confidence Score</div>
                </div>
                <div className="text-center">
                  <div className={`text-3xl font-bold ${isMatch ? "text-green-600" : "text-red-600"}`}>
                    {isMatch ? "MATCH" : "NO MATCH"}
                  </div>
                  <div className="text-sm text-gray-600">Verification Result</div>
                </div>
              </div>

              {/* Image Comparison */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="text-center">
                  <h3 className="font-medium mb-2">Captured Selfie</h3>
                  <img
                    src={selfieData.imageUrl || "/placeholder.svg"}
                    alt="Captured selfie"
                    className="w-full h-48 object-cover rounded-lg border"
                  />
                </div>
                <div className="text-center">
                  <h3 className="font-medium mb-2">Reference Image</h3>
                  <img
                    src={referencePreview || "/placeholder.svg"}
                    alt="Reference"
                    className="w-full h-48 object-cover rounded-lg border"
                  />
                </div>
              </div>

              {/* Additional Details */}
              {verificationResult.details && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Verification Details</h3>
                  <pre className="text-sm text-gray-600 whitespace-pre-wrap">
                    {JSON.stringify(verificationResult.details, null, 2)}
                  </pre>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-center space-x-4">
                <Button onClick={resetVerification} variant="outline">
                  Verify Another
                </Button>
                <Button onClick={() => router.push("/capture")}>New Selfie</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Face Verification</h1>
          <p className="text-gray-600">Upload a reference image to compare with your selfie</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Captured Selfie */}
          <Card>
            <CardHeader>
              <CardTitle>Captured Selfie</CardTitle>
              <CardDescription>Your live selfie for verification</CardDescription>
            </CardHeader>
            <CardContent>
              <img
                src={selfieData.imageUrl || "/placeholder.svg"}
                alt="Captured selfie"
                className="w-full h-64 object-cover rounded-lg border"
              />
            </CardContent>
          </Card>

          {/* Reference Image Upload */}
          <Card>
            <CardHeader>
              <CardTitle>Reference Image</CardTitle>
              <CardDescription>Upload an image to compare against</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="reference-upload">Upload Reference Image</Label>
                <Input
                  ref={fileInputRef}
                  id="reference-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleReferenceUpload}
                  className="mt-2"
                />
              </div>

              {referencePreview ? (
                <img
                  src={referencePreview || "/placeholder.svg"}
                  alt="Reference preview"
                  className="w-full h-64 object-cover rounded-lg border"
                />
              ) : (
                <div className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <Upload className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                    <p className="text-gray-500">Upload reference image</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Verification Controls */}
        <Card className="mt-6">
          <CardContent className="pt-6 space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {isVerifying && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Verifying faces...</span>
                  <span>{progress}%</span>
                </div>
                <Progress value={progress} />
              </div>
            )}

            <Button
              onClick={performVerification}
              disabled={!referenceImage || isVerifying}
              className="w-full"
              size="lg"
            >
              {isVerifying ? "Verifying..." : "Verify Faces"}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
