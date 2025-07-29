"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileText, X, AlertCircle } from "lucide-react"
import { useRouter } from "next/navigation"

interface UploadedFile {
  file: File
  preview: string
  type: string
}

export default function UploadPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [documentType, setDocumentType] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState("")
  const router = useRouter()

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files) return

    Array.from(files).forEach((file) => {
      if (file.type.startsWith("image/") || file.type === "application/pdf") {
        const reader = new FileReader()
        reader.onload = (e) => {
          const newFile: UploadedFile = {
            file,
            preview: e.target?.result as string,
            type: file.type,
          }
          setUploadedFiles((prev) => [...prev, newFile])
        }
        reader.readAsDataURL(file)
      }
    })
  }, [])

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if (uploadedFiles.length === 0 || !documentType) {
      setError("Please upload at least one document and select document type")
      return
    }

    setIsProcessing(true)
    setError("")
    setProgress(0)

    try {
      const formData = new FormData()
      uploadedFiles.forEach((uploadedFile, index) => {
        formData.append(`files`, uploadedFile.file)
      })
      formData.append("document_type", documentType)

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 500)

      // Replace with your actual FastAPI endpoint
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/documents/upload`,
        {
          method: "POST",
          body: formData,
        },
      )

      clearInterval(progressInterval)
      setProgress(100)

      if (!response.ok) {
        throw new Error("Upload failed")
      }

      const result = await response.json()

      // Redirect to results page with the processing ID
      setTimeout(() => {
        router.push(`/results?id=${result.processing_id}`)
      }, 1000)
    } catch (err) {
      setError("Upload failed. Please try again.")
      setIsProcessing(false)
      setProgress(0)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Document Upload</h1>
          <p className="text-gray-600">Upload your identity documents for KYC verification</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Documents</CardTitle>
              <CardDescription>Supported formats: JPG, PNG, PDF (Max 10MB per file)</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <Label htmlFor="document-type">Document Type</Label>
                <Select value={documentType} onValueChange={setDocumentType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select document type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="passport">Passport</SelectItem>
                    <SelectItem value="drivers_license">Driver's License</SelectItem>
                    <SelectItem value="national_id">National ID</SelectItem>
                    <SelectItem value="utility_bill">Utility Bill</SelectItem>
                    <SelectItem value="bank_statement">Bank Statement</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="file-upload">Upload Files</Label>
                <div className="mt-2">
                  <Input
                    id="file-upload"
                    type="file"
                    multiple
                    accept="image/*,.pdf"
                    onChange={handleFileUpload}
                    className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {isProcessing && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Processing documents...</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} />
                </div>
              )}

              <Button
                onClick={handleSubmit}
                disabled={uploadedFiles.length === 0 || !documentType || isProcessing}
                className="w-full"
              >
                {isProcessing ? "Processing..." : "Start Verification"}
              </Button>
            </CardContent>
          </Card>

          {/* Preview Section */}
          <Card>
            <CardHeader>
              <CardTitle>Uploaded Files</CardTitle>
              <CardDescription>{uploadedFiles.length} file(s) ready for processing</CardDescription>
            </CardHeader>
            <CardContent>
              {uploadedFiles.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No files uploaded yet</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {uploadedFiles.map((uploadedFile, index) => (
                    <div key={index} className="flex items-center space-x-4 p-3 border rounded-lg">
                      <div className="flex-shrink-0">
                        {uploadedFile.type.startsWith("image/") ? (
                          <img
                            src={uploadedFile.preview || "/placeholder.svg"}
                            alt="Preview"
                            className="w-12 h-12 object-cover rounded"
                          />
                        ) : (
                          <FileText className="w-12 h-12 text-red-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">{uploadedFile.file.name}</p>
                        <p className="text-sm text-gray-500">{(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                      <Button variant="ghost" size="sm" onClick={() => removeFile(index)} disabled={isProcessing}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
