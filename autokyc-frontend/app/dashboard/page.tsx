"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { FileText, CheckCircle, XCircle, Clock, TrendingUp } from "lucide-react"

interface DashboardStats {
  total_verifications: number
  successful_verifications: number
  pending_verifications: number
  failed_verifications: number
  success_rate: number
}

interface RecentVerification {
  id: string
  verification_type: string
  status: string
  created_at: string
  confidence_score: number
}

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats>({
    total_verifications: 0,
    successful_verifications: 0,
    pending_verifications: 0,
    failed_verifications: 0,
    success_rate: 0,
  })
  const [recentVerifications, setRecentVerifications] = useState<RecentVerification[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Mock data - replace with actual API calls
    setTimeout(() => {
      setStats({
        total_verifications: 1247,
        successful_verifications: 1156,
        pending_verifications: 23,
        failed_verifications: 68,
        success_rate: 92.7,
      })

      setRecentVerifications([
        {
          id: "doc_001",
          verification_type: "selfie_match",
          status: "completed",
          created_at: "2024-01-15T10:30:00Z",
          confidence_score: 95.8,
        },
        {
          id: "doc_002",
          verification_type: "id_verification",
          status: "completed",
          created_at: "2024-01-15T09:15:00Z",
          confidence_score: 88.2,
        },
        {
          id: "doc_003",
          verification_type: "biometric_check",
          status: "processing",
          created_at: "2024-01-15T08:45:00Z",
          confidence_score: 0,
        },
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case "failed":
        return <XCircle className="w-4 h-4 text-red-600" />
      default:
        return <Clock className="w-4 h-4 text-yellow-600" />
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Clock className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p>Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
          <p className="text-gray-600">Overview of your KYC verification activities</p>
        </div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Verifications</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_verifications.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">All time</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Successful</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.successful_verifications.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">Successfully verified</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Pending</CardTitle>
              <Clock className="h-4 w-4 text-yellow-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{stats.pending_verifications}</div>
              <p className="text-xs text-muted-foreground">In processing</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <TrendingUp className="h-4 w-4 text-blue-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{stats.success_rate}%</div>
              <p className="text-xs text-muted-foreground">Verification success</p>
            </CardContent>
          </Card>
        </div>

        {/* Recent Verifications */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Verifications</CardTitle>
            <CardDescription>Latest document verification activities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentVerifications.map((verification) => (
                <div key={verification.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(verification.status)}
                    <div>
                      <div className="font-medium">
                        {verification.verification_type.replace("_", " ").toUpperCase()}
                      </div>
                      <div className="text-sm text-gray-500">ID: {verification.id}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    {verification.confidence_score > 0 && (
                      <div className="text-sm">
                        <span className="font-medium">{verification.confidence_score}%</span>
                        <span className="text-gray-500"> confidence</span>
                      </div>
                    )}
                    <Badge className={getStatusColor(verification.status)}>{verification.status}</Badge>
                    <div className="text-sm text-gray-500">
                      {new Date(verification.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
