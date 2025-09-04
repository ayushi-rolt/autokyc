"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, AlertTriangle, XCircle, Download, Home, RefreshCw, Clock } from "lucide-react"
import Link from "next/link"

type VerificationStatus = "processing" | "success" | "flagged" | "failed"

export default function KYCResultsPage() {
  const router = useRouter()
  const [status, setStatus] = useState<VerificationStatus>("processing")
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    // Simulate verification process
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          // Randomly set final status for demo
          const outcomes: VerificationStatus[] = ["success", "flagged", "failed"]
          setStatus(outcomes[Math.floor(Math.random() * outcomes.length)])
          return 100
        }
        return prev + 10
      })
    }, 500)

    return () => clearInterval(interval)
  }, [])

  const getStatusConfig = () => {
    switch (status) {
      case "processing":
        return {
          icon: RefreshCw,
          title: "Processing Your KYC",
          description: "Please wait while we verify your information...",
          color: "text-blue-600",
          bgColor: "bg-blue-50",
          borderColor: "border-blue-200",
        }
      case "success":
        return {
          icon: CheckCircle,
          title: "KYC Verification Successful",
          description: "Your identity has been successfully verified.",
          color: "text-green-600",
          bgColor: "bg-green-50",
          borderColor: "border-green-200",
        }
      case "flagged":
        return {
          icon: AlertTriangle,
          title: "KYC Under Review",
          description: "Your application requires manual review. You will be contacted within 24 hours.",
          color: "text-yellow-600",
          bgColor: "bg-yellow-50",
          borderColor: "border-yellow-200",
        }
      case "failed":
        return {
          icon: XCircle,
          title: "KYC Verification Failed",
          description: "We could not verify your identity. Please try again or contact support.",
          color: "text-red-600",
          bgColor: "bg-red-50",
          borderColor: "border-red-200",
        }
    }
  }

  const statusConfig = getStatusConfig()
  const StatusIcon = statusConfig.icon

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Step 4 of 4</span>
            <span className="text-sm text-gray-500">Verification Results</span>
          </div>
          <Progress value={100} className="h-2" />
        </div>

        <div className="space-y-8">
          {/* Status Card */}
          <Card className={`${statusConfig.bgColor} ${statusConfig.borderColor} border-2`}>
            <CardContent className="pt-8 pb-8">
              <div className="text-center">
                <StatusIcon
                  className={`h-16 w-16 ${statusConfig.color} mx-auto mb-4 ${status === "processing" ? "animate-spin" : ""}`}
                />
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">{statusConfig.title}</h1>
                <p className="text-gray-600 dark:text-gray-300 text-lg">{statusConfig.description}</p>

                {status === "processing" && (
                  <div className="mt-6">
                    <Progress value={progress} className="h-3 max-w-md mx-auto" />
                    <p className="text-sm text-gray-500 mt-2">{progress}% Complete</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {status !== "processing" && (
            <>
              {/* Verification Details */}
              <Card>
                <CardHeader>
                  <CardTitle>Verification Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className="flex items-center justify-center mb-2">
                        {status === "success" ? (
                          <CheckCircle className="h-8 w-8 text-green-500" />
                        ) : (
                          <XCircle className="h-8 w-8 text-red-500" />
                        )}
                      </div>
                      <h3 className="font-medium">Face Verification</h3>
                      <Badge variant={status === "success" ? "default" : "destructive"} className="mt-1">
                        {status === "success" ? "Passed" : "Failed"}
                      </Badge>
                    </div>

                    <div className="text-center">
                      <div className="flex items-center justify-center mb-2">
                        {status === "success" ? (
                          <CheckCircle className="h-8 w-8 text-green-500" />
                        ) : (
                          <XCircle className="h-8 w-8 text-red-500" />
                        )}
                      </div>
                      <h3 className="font-medium">Document OCR</h3>
                      <Badge variant={status === "success" ? "default" : "destructive"} className="mt-1">
                        {status === "success" ? "Passed" : "Failed"}
                      </Badge>
                    </div>

                    <div className="text-center">
                      <div className="flex items-center justify-center mb-2">
                        {status === "success" ? (
                          <CheckCircle className="h-8 w-8 text-green-500" />
                        ) : status === "flagged" ? (
                          <AlertTriangle className="h-8 w-8 text-yellow-500" />
                        ) : (
                          <XCircle className="h-8 w-8 text-red-500" />
                        )}
                      </div>
                      <h3 className="font-medium">Liveness Check</h3>
                      <Badge
                        variant={status === "success" ? "default" : status === "flagged" ? "secondary" : "destructive"}
                        className="mt-1"
                      >
                        {status === "success" ? "Passed" : status === "flagged" ? "Review" : "Failed"}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Session Information */}
              <Card>
                <CardHeader>
                  <CardTitle>Session Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Session ID:</span>
                      <p className="font-mono">KYC-{Date.now().toString().slice(-8)}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Verification Date:</span>
                      <p>{new Date().toLocaleDateString()}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Processing Time:</span>
                      <p>2 minutes 34 seconds</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Reference Number:</span>
                      <p className="font-mono">REF-{Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                {status === "success" && (
                  <Button size="lg" className="px-8">
                    <Download className="h-4 w-4 mr-2" />
                    Download Certificate
                  </Button>
                )}

                {status === "failed" && (
                  <Button size="lg" className="px-8" onClick={() => router.push("/kyc/start")}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry Verification
                  </Button>
                )}

                <Link href="/">
                  <Button variant="outline" size="lg" className="px-8 bg-transparent">
                    <Home className="h-4 w-4 mr-2" />
                    Back to Home
                  </Button>
                </Link>
              </div>

              {status === "flagged" && (
                <Card className="bg-yellow-50 border-yellow-200">
                  <CardContent className="pt-6">
                    <div className="flex items-start space-x-3">
                      <Clock className="h-5 w-5 text-yellow-600 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-yellow-900">What happens next?</h4>
                        <p className="text-sm text-yellow-800 mt-1">
                          Our verification team will manually review your application. You will receive an email
                          notification within 24 hours with the final decision. If you have any questions, please
                          contact our support team.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}
