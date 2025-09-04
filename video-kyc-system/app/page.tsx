import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Clock, Users, FileText, CheckCircle, Video, Shield } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Fingerprint } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Fingerprint className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">SecureKYC</span>
          </div>
          <nav className="hidden md:flex items-center space-x-6">
            <Link
              href="/admin"
              className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400"
            >
              Admin Dashboard
            </Link>
            <Link
              href="/officer"
              className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400"
            >
              Officer Portal
            </Link>
            <ThemeToggle />
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">Automated Video KYC</h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Complete your Know Your Customer verification in minutes with our secure, AI-powered video verification
            system.
          </p>
          <Link href="/kyc/start">
            <Button size="lg" className="text-lg px-8 py-4">
              Start KYC Verification
            </Button>
          </Link>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card className="text-center">
            <CardHeader>
              <Video className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <CardTitle>Video Verification</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Real-time liveness detection and face matching using advanced AI algorithms
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <FileText className="h-12 w-12 text-green-600 mx-auto mb-4" />
              <CardTitle>Document OCR</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>Automatic extraction and verification of Aadhaar and PAN card details</CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <CheckCircle className="h-12 w-12 text-purple-600 mx-auto mb-4" />
              <CardTitle>Instant Results</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>Get verification results within minutes with detailed audit trails</CardDescription>
            </CardContent>
          </Card>
        </div>

        {/* Stats Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="flex items-center justify-center mb-2">
                <Clock className="h-6 w-6 text-blue-600 mr-2" />
                <span className="text-3xl font-bold text-gray-900 dark:text-white">2 min</span>
              </div>
              <p className="text-gray-600 dark:text-gray-400">Average completion time</p>
            </div>
            <div>
              <div className="flex items-center justify-center mb-2">
                <Shield className="h-6 w-6 text-green-600 mr-2" />
                <span className="text-3xl font-bold text-gray-900">99.8%</span>
              </div>
              <p className="text-gray-600">Accuracy rate</p>
            </div>
            <div>
              <div className="flex items-center justify-center mb-2">
                <Users className="h-6 w-6 text-purple-600 mr-2" />
                <span className="text-3xl font-bold text-gray-900">50K+</span>
              </div>
              <p className="text-gray-600">Verifications completed</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
