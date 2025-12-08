"use client"

import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Camera, FileText, CheckCircle, ArrowLeft, ArrowRight, Scan, AlertCircle } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

// Ensure this matches your FastAPI port
const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

export default function DocumentUploadPage() {
  const router = useRouter()
  const aadhaarInputRef = useRef<HTMLInputElement>(null)
  const panInputRef = useRef<HTMLInputElement>(null)
  
  const [documents, setDocuments] = useState({
    aadhaar: null as File | null,
    pan: null as File | null,
  })
  
  const [ocrResults, setOcrResults] = useState({
    aadhaar: null as any,
    pan: null as any,
  })
  
  const [processing, setProcessing] = useState({
    aadhaar: false,
    pan: false,
  })
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 1. Handle individual file upload and preview (OCR)
  const handleFileUpload = async (type: "aadhaar" | "pan", file: File) => {
    // Update state to show we have the file
    setDocuments((prev) => ({ ...prev, [type]: file }))
    setProcessing((prev) => ({ ...prev, [type]: true }))
    setError(null)

    try {
      // Create FormData to send to FastAPI for preview/verification
      const formData = new FormData()
      formData.append("file", file) // FastAPI's /verify-document/ expects 'file'

      // Call the Single Document Verification Endpoint
      const response = await fetch(`${FASTAPI_URL}/verify-document/`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to process document")
      }

      const data = await response.json()
      
      // Map the Backend Data to your Frontend UI State
      const fields = data.fields || {}
      let formattedResult;

      if (type === "aadhaar") {
        formattedResult = {
          name: fields.name || "Not detected",
          aadhaarNumber: fields.aadhaar_number || "Not detected",
          dob: fields.dob || "Not detected",
          address: fields.address || "Address extraction not supported yet", 
        }
      } else {
        formattedResult = {
          name: fields.name || "Not detected",
          panNumber: fields.pan_number || "Not detected",
          dob: fields.dob || "Not detected",
        }
      }

      setOcrResults((prev) => ({ ...prev, [type]: formattedResult }))

    } catch (err) {
      console.error("OCR Error:", err)
      setError(`Failed to process ${type}. Please try a clearer photo.`)
    } finally {
      setProcessing((prev) => ({ ...prev, [type]: false }))
    }
  }

  // 2. Camera Capture Handler
  const captureDocument = async (type: "aadhaar" | "pan") => {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true })
      if (type === "aadhaar") {
        aadhaarInputRef.current?.click()
      } else {
        panInputRef.current?.click()
      }
    } catch (error) {
      console.error("Camera access denied:", error)
      setError("Camera access denied. Please use 'Choose File' instead.")
    }
  }

  // 3. Final Submission Handler
  const proceedToResults = async () => {
    // Validation: Ensure at least ONE document is uploaded
    if (!documents.aadhaar && !documents.pan) {
        setError("Please upload at least one document (Aadhaar or PAN) to continue.")
        return
    }

    setLoading(true)
    setError(null)

    try {
      const userId = sessionStorage.getItem("userId")
      const finalUserId = userId || "test-user-123" 

      const formDataToSend = new FormData()
      formDataToSend.append("user_id", finalUserId)

      // Only append the file if it exists
      if (documents.aadhaar) {
          formDataToSend.append("aadhaar", documents.aadhaar)
      }
      if (documents.pan) {
          formDataToSend.append("pan", documents.pan)
      }

      // This calls your Next.js API route, which forwards to FastAPI
      const response = await fetch("/api/kyc/submit-documents", {
        method: "POST",
        body: formDataToSend,
      })

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(errText || "Failed to upload documents")
      }

      router.push("/kyc/results")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during final submission")
      console.error("Error uploading documents:", err)
    } finally {
      setLoading(false)
    }
  }

  // 4. Logic Check: Unlock button if (Aadhaar is Ready) OR (Pan is Ready)
  const isAadhaarReady = documents.aadhaar && ocrResults.aadhaar
  const isPanReady = documents.pan && ocrResults.pan
  const canProceed = isAadhaarReady || isPanReady

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link
            href="/kyc/video-capture"
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
            <span className="text-sm font-medium">Step 2 of 4</span>
            <span className="text-sm text-gray-500">Document Upload</span>
          </div>
          <Progress value={50} className="h-2" />
        </div>

        <div className="space-y-8">
          <div className="text-center">
            <FileText className="h-16 w-16 text-blue-600 mx-auto mb-4" />
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Upload Identity Documents</h1>
            
            <div className="flex items-center justify-center gap-2 text-gray-600 dark:text-gray-300">
                <AlertCircle className="h-4 w-4" />
                <p>Please upload either your <strong>Aadhaar Card</strong> OR <strong>PAN Card</strong></p>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm">{error}</div>
          )}

          <Tabs defaultValue="aadhaar" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="aadhaar" className="flex items-center space-x-2">
                <span>Aadhaar Card</span>
                {ocrResults.aadhaar && <CheckCircle className="h-4 w-4 text-green-500" />}
              </TabsTrigger>
              <TabsTrigger value="pan" className="flex items-center space-x-2">
                <span>PAN Card</span>
                {ocrResults.pan && <CheckCircle className="h-4 w-4 text-green-500" />}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="aadhaar">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <FileText className="h-5 w-5" />
                    <span>Aadhaar Card Upload</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {!documents.aadhaar ? (
                    <div className="space-y-4">
                      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                        <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600 mb-4">Upload your Aadhaar card image</p>
                        <div className="flex justify-center space-x-4">
                          <Button onClick={() => aadhaarInputRef.current?.click()}>
                            <Upload className="h-4 w-4 mr-2" />
                            Choose File
                          </Button>
                          <Button variant="outline" onClick={() => captureDocument("aadhaar")}>
                            <Camera className="h-4 w-4 mr-2" />
                            Take Photo
                          </Button>
                        </div>
                        <input
                          ref={aadhaarInputRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0]
                            if (file) handleFileUpload("aadhaar", file)
                          }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex items-center space-x-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <span className="text-green-800">Aadhaar card uploaded successfully</span>
                        {/* Option to clear and re-upload */}
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="ml-auto text-red-500 hover:text-red-700 hover:bg-red-50"
                          onClick={() => {
                             setDocuments(prev => ({...prev, aadhaar: null}));
                             setOcrResults(prev => ({...prev, aadhaar: null}));
                          }}
                        >
                          Change
                        </Button>
                      </div>

                      {processing.aadhaar ? (
                        <div className="flex items-center space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <Scan className="h-5 w-5 text-blue-600 animate-spin" />
                          <span className="text-blue-800">Processing document...</span>
                        </div>
                      ) : (
                        ocrResults.aadhaar && (
                          <div className="p-4 bg-gray-50 border rounded-lg">
                            <h4 className="font-medium mb-3">Extracted Information:</h4>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-600">Name:</span>
                                <p className="font-medium">{ocrResults.aadhaar.name}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">Aadhaar Number:</span>
                                <p className="font-medium">{ocrResults.aadhaar.aadhaarNumber}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">Date of Birth:</span>
                                <p className="font-medium">{ocrResults.aadhaar.dob}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">Address:</span>
                                <p className="font-medium">{ocrResults.aadhaar.address}</p>
                              </div>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="pan">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <FileText className="h-5 w-5" />
                    <span>PAN Card Upload</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {!documents.pan ? (
                    <div className="space-y-4">
                      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                        <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600 mb-4">Upload your PAN card image</p>
                        <div className="flex justify-center space-x-4">
                          <Button onClick={() => panInputRef.current?.click()}>
                            <Upload className="h-4 w-4 mr-2" />
                            Choose File
                          </Button>
                          <Button variant="outline" onClick={() => captureDocument("pan")}>
                            <Camera className="h-4 w-4 mr-2" />
                            Take Photo
                          </Button>
                        </div>
                        <input
                          ref={panInputRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0]
                            if (file) handleFileUpload("pan", file)
                          }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex items-center space-x-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <span className="text-green-800">PAN card uploaded successfully</span>
                        {/* Option to clear and re-upload */}
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="ml-auto text-red-500 hover:text-red-700 hover:bg-red-50"
                          onClick={() => {
                             setDocuments(prev => ({...prev, pan: null}));
                             setOcrResults(prev => ({...prev, pan: null}));
                          }}
                        >
                          Change
                        </Button>
                      </div>

                      {processing.pan ? (
                        <div className="flex items-center space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <Scan className="h-5 w-5 text-blue-600 animate-spin" />
                          <span className="text-blue-800">Processing document...</span>
                        </div>
                      ) : (
                        ocrResults.pan && (
                          <div className="p-4 bg-gray-50 border rounded-lg">
                            <h4 className="font-medium mb-3">Extracted Information:</h4>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-600">Name:</span>
                                <p className="font-medium">{ocrResults.pan.name}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">PAN Number:</span>
                                <p className="font-medium">{ocrResults.pan.panNumber}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">Date of Birth:</span>
                                <p className="font-medium">{ocrResults.pan.dob}</p>
                              </div>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {canProceed && (
            <div className="text-center">
              <Button onClick={proceedToResults} size="lg" className="px-8" disabled={loading}>
                {loading ? "Uploading..." : "Continue to Verification"}
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}