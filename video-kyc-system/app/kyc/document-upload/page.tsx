"use client"

import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Camera, FileText, CheckCircle, ArrowLeft, ArrowRight, Scan, AlertCircle } from "lucide-react" // Added AlertCircle
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

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

  // --- 1. MODIFIED: Added error state ---
  const [errors, setErrors] = useState({
    aadhaar: null as string | null,
    pan: null as string | null,
  })

  // --- 2. REPLACED: Mock function with real API call ---
  const handleFileUpload = async (type: "aadhaar" | "pan", file: File) => {
    setDocuments((prev) => ({ ...prev, [type]: file }))
    setProcessing((prev) => ({ ...prev, [type]: true }))
    setOcrResults((prev) => ({ ...prev, [type]: null })) // Clear previous results
    setErrors((prev) => ({ ...prev, [type]: null })) // Clear previous errors

    // FormData is used to send files
    const formData = new FormData()
    // The key "file" MUST match your FastAPI argument name:
    // async def verify_document(file: UploadFile = File(...)):
    formData.append("file", file)

    try {
      // Make the fetch request to your running FastAPI server
      const response = await fetch("http://127.0.0.1:8000/verify-document/", {
        method: "POST",
        body: formData,
      })

      const data = await response.json() // Expects { verified, fields, raw_text }

      if (!response.ok) {
        throw new Error(data.error || "Server error, please try again.")
      }
      if (!data.verified || !data.fields || Object.keys(data.fields).length === 0) {
        throw new Error("Could not extract any fields. Please try a clearer image.")
      }

      // **IMPORTANT: Data Transformation**
      // Your API returns { fields: { pan_number: "..." } }
      // Your UI expects { panNumber: "..." }
      // We must map the snake_case (API) to camelCase (UI)
      
      const apiFields = data.fields
      let transformedResult: any = {}

      if (type === "aadhaar") {
        transformedResult = {
          name: apiFields.name || "N/A",
          aadhaarNumber: apiFields.aadhaar_number || "N/A", // Map snake_case
          dob: apiFields.dob || "N/A",
          address: "N/A", // Your API doesn't extract address, so we default
        }
      } else {
        transformedResult = {
          name: apiFields.name || "N/A",
          panNumber: apiFields.pan_number || "N/A", // Map snake_case
          dob: apiFields.dob || "N/A",
        }
      }
      
      setOcrResults((prev) => ({ ...prev, [type]: transformedResult }))

    } catch (err: any) {
      console.error("OCR processing failed:", err)
      setErrors((prev) => ({ ...prev, [type]: err.message }))
    } finally {
      setProcessing((prev) => ({ ...prev, [type]: false }))
    }
  }

  const captureDocument = async (type: "aadhaar" | "pan") => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      // In a real implementation, you would show a camera interface
      // For now, we'll simulate file selection
      if (type === "aadhaar") {
        aadhaarInputRef.current?.click()
      } else {
        panInputRef.current?.click()
      }
    } catch (error) {
      console.error("Camera access denied:", error)
    }
  }

  const proceedToResults = () => {
    if (documents.aadhaar && documents.pan && ocrResults.aadhaar && ocrResults.pan) {
      router.push("/kyc/results")
    }
  }

  const canProceed = documents.aadhaar && documents.pan && ocrResults.aadhaar && ocrResults.pan

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
            <p className="text-gray-600 dark:text-gray-300">Please upload clear images of your Aadhaar and PAN cards</p>
          </div>

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
                      </div>

                      {/* --- 3. ADDED: Error message display --- */}
                      {errors.aadhaar && (
                        <div className="flex items-center space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                          <AlertCircle className="h-5 w-5 text-red-600" />
                          <span className="text-red-800">{errors.aadhaar}</span>
                        </div>
                      )}
                      
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
                      </div>

                      {/* --- 3. ADDED: Error message display --- */}
                      {errors.pan && (
                        <div className="flex items-center space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                          <AlertCircle className="h-5 w-5 text-red-600" />
                          <span className="text-red-800">{errors.pan}</span>
                        </div>
                      )}

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
              <Button onClick={proceedToResults} size="lg" className="px-8">
                Continue to Verification
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}