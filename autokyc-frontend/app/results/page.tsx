"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, XCircle, Clock, FileText, User, Calendar, MapPin, Download } from "lucide-react"
import { useSearchParams } from "next/navigation"
import Link from "next/link"

interface VerificationResult {
  id: string
  status: "processing" | "completed" | "failed"
  document_type: string
  extracted_data: {
    name?: string
    document_number?: string
    date_of_birth?: string
    expiry_date?: string
    address?: string
    nationality?: string
  }
  verification_checks: {
    document_authenticity: boolean
    data_consistency: boolean
    expiry_check: boolean
    format_validation: boolean
  }
  confidence_score: number
  processed_at: string
}

export default function ResultsPage() {
  const searchParams = useSearchParams()
  const processingId = searchParams.get("id")
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    const fetchResults = async () => {
      if (!processingId) {
        setError("No processing ID provided")
        setLoading(false)
        return
      }

      try {
        // Replace with your actual FastAPI endpoint
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/documents/results/${processingId}`,
        )

        if (!response.ok) {
          throw new Error("Failed to fetch results")
        }

        const data = await response.json()
        setResult(data)
      } catch (err) {
        setError("Failed to load verification results")
        // Mock data for demonstration
        setResult({
          id: processingId || "demo",
          status: "completed",
          document_type: "passport",
          extracted_data: {
            name: "John Doe",
            document_number: "P123456789",
            date_of_birth: "1990-01-15",
            expiry_date: "2030-01-15",
            address: "123 Main St, City, Country",
            nationality: "US",
          },
          verification_checks: {
            document_authenticity: true,
            data_consistency: true,
            expiry_check: true,
            format_validation: true,
          },
          confidence_score: 95.8,
          processed_at: new Date().toISOString(),
        })
      } finally {
        setLoading(false)
      }
    }

    fetchResults()
  }, [processingId])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col items-center py-8">
            <Clock className="w-12 h-12 text-blue-600 animate-spin mb-4" />
            <h3 className="text-lg font-semibold mb-2">Processing Document</h3>
            <p className="text-gray-600 text-center">Please wait while we verify your document...</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (error && !result) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col items-center py-8">
            <XCircle className="w-12 h-12 text-red-600 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Error</h3>
            <p className="text-gray-600 text-center mb-4">{error}</p>
            <Link href="/upload">
              <Button>Try Again</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!result) return null

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-6 h-6 text-green-600" />
      case "failed":
        return <XCircle className="w-6 h-6 text-red-600" />
      default:
        return <Clock className="w-6 h-6 text-yellow-600" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800"
      case "failed":
        return "bg-red-100 text-red-800"
      default:
        return "bg-yellow-100 text-yellow-800"
    }
  }

  const allChecksPassed = Object.values(result.verification_checks).every((check) => check)

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Verification Results</h1>
          <p className="text-gray-600">Document processing completed</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Status Overview */}
          <Card className="lg:col-span-3">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(result.status)}
                  <div>
                    <CardTitle>Verification Status</CardTitle>
                    <CardDescription>Processing ID: {result.id}</CardDescription>
                  </div>
                </div>
                <Badge className={getStatusColor(result.status)}>{result.status.toUpperCase()}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{result.confidence_score}%</div>
                  <div className="text-sm text-gray-600">Confidence Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {Object.values(result.verification_checks).filter(Boolean).length}/4
                  </div>
                  <div className="text-sm text-gray-600">Checks Passed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{result.document_type.toUpperCase()}</div>
                  <div className="text-sm text-gray-600">Document Type</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Extracted Data */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="w-5 h-5" />
                <span>Extracted Information</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {result.extracted_data.name && (
                <div className="flex items-center space-x-3">
                  <User className="w-4 h-4 text-gray-500" />
                  <div>
                    <div className="font-medium">Full Name</div>
                    <div className="text-gray-600">{result.extracted_data.name}</div>
                  </div>
                </div>
              )}

              {result.extracted_data.document_number && (
                <div className="flex items-center space-x-3">
                  <FileText className="w-4 h-4 text-gray-500" />
                  <div>
                    <div className="font-medium">Document Number</div>
                    <div className="text-gray-600">{result.extracted_data.document_number}</div>
                  </div>
                </div>
              )}

              {result.extracted_data.date_of_birth && (
                <div className="flex items-center space-x-3">
                  <Calendar className="w-4 h-4 text-gray-500" />
                  <div>
                    <div className="font-medium">Date of Birth</div>
                    <div className="text-gray-600">{result.extracted_data.date_of_birth}</div>
                  </div>
                </div>
              )}

              {result.extracted_data.address && (
                <div className="flex items-center space-x-3">
                  <MapPin className="w-4 h-4 text-gray-500" />
                  <div>
                    <div className="font-medium">Address</div>
                    <div className="text-gray-600">{result.extracted_data.address}</div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Verification Checks */}
          <Card>
            <CardHeader>
              <CardTitle>Verification Checks</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(result.verification_checks).map(([check, passed]) => (
                <div key={check} className="flex items-center justify-between">
                  <span className="text-sm capitalize">{check.replace("_", " ")}</span>
                  {passed ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600" />
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Actions */}
        <div className="mt-8 flex flex-wrap gap-4 justify-center">
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </Button>
          <Link href="/upload">
            <Button>Verify Another Document</Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
