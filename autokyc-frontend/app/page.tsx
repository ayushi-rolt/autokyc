"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { CheckCircle, Camera, User } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">AutoKYC Face Verification Platform</h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Automated identity verification using advanced facial recognition and biometric matching
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Card className="text-center hover:shadow-lg transition-shadow">
            <CardHeader>
              <Camera className="w-12 h-12 mx-auto text-blue-600 mb-4" />
              <CardTitle>Capture Selfie</CardTitle>
              <CardDescription>Take a live selfie using your device camera</CardDescription>
            </CardHeader>
          </Card>

          <Card className="text-center hover:shadow-lg transition-shadow">
            <CardHeader>
              <User className="w-12 h-12 mx-auto text-green-600 mb-4" />
              <CardTitle>Face Analysis</CardTitle>
              <CardDescription>Advanced facial feature extraction and embedding generation</CardDescription>
            </CardHeader>
          </Card>

          <Card className="text-center hover:shadow-lg transition-shadow">
            <CardHeader>
              <CheckCircle className="w-12 h-12 mx-auto text-purple-600 mb-4" />
              <CardTitle>Biometric Matching</CardTitle>
              <CardDescription>Real-time face comparison and identity verification</CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* CTA Section */}
        <div className="text-center">
          <Card className="max-w-md mx-auto">
            <CardHeader>
              <CardTitle>Start Face Verification</CardTitle>
              <CardDescription>Verify your identity using facial recognition in seconds</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/capture">
                <Button size="lg" className="w-full">
                  Capture Selfie
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
