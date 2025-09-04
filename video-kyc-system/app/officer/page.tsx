"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertTriangle, Search, Eye, Play, Download, MessageSquare, Clock, Fingerprint } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"

// Mock data for officer dashboard
const flaggedSessions = [
  {
    id: "KYC-12345679",
    name: "Jane Smith",
    email: "jane@example.com",
    status: "flagged",
    date: "2024-01-15",
    time: "13:45",
    reason: "Face match confidence below threshold",
    priority: "high",
  },
  {
    id: "KYC-12345682",
    name: "Mike Wilson",
    email: "mike@example.com",
    status: "flagged",
    date: "2024-01-15",
    time: "10:20",
    reason: "Document quality issues",
    priority: "medium",
  },
  {
    id: "KYC-12345683",
    name: "Sarah Davis",
    email: "sarah@example.com",
    status: "flagged",
    date: "2024-01-14",
    time: "16:30",
    reason: "Liveness detection uncertain",
    priority: "low",
  },
]

export default function OfficerDashboard() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedSession, setSelectedSession] = useState<string | null>(null)

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "high":
        return <Badge variant="destructive">High Priority</Badge>
      case "medium":
        return <Badge variant="secondary">Medium Priority</Badge>
      case "low":
        return <Badge variant="outline">Low Priority</Badge>
      default:
        return <Badge variant="outline">Unknown</Badge>
    }
  }

  const filteredSessions = flaggedSessions.filter(
    (session) =>
      session.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.id.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="bg-white dark:bg-gray-800 border-b dark:border-gray-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Fingerprint className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Bank Officer Portal</h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">KYC Review & Verification</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="text-red-600 border-red-200 dark:text-red-400 dark:border-red-800">
                <AlertTriangle className="h-3 w-3 mr-1" />
                {flaggedSessions.length} Pending Reviews
              </Badge>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="flagged" className="space-y-6">
          <TabsList>
            <TabsTrigger value="flagged">Flagged Sessions</TabsTrigger>
            <TabsTrigger value="review">Manual Review</TabsTrigger>
            <TabsTrigger value="audit">Audit Log</TabsTrigger>
          </TabsList>

          <TabsContent value="flagged" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Sessions Requiring Review</CardTitle>
                <CardDescription>KYC sessions that have been flagged for manual verification</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-6">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search flagged sessions..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  {filteredSessions.map((session) => (
                    <div key={session.id} className="border rounded-lg p-4 hover:bg-gray-50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div>
                            <p className="font-medium">{session.name}</p>
                            <p className="text-sm text-gray-500">{session.email}</p>
                            <p className="text-xs text-gray-400">{session.id}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className="text-right">
                            <p className="text-sm font-medium">{session.date}</p>
                            <p className="text-sm text-gray-500">{session.time}</p>
                          </div>
                          {getPriorityBadge(session.priority)}
                        </div>
                      </div>

                      <div className="mt-3 pt-3 border-t">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-600">Flagged Reason:</p>
                            <p className="text-sm font-medium text-red-600">{session.reason}</p>
                          </div>
                          <div className="flex space-x-2">
                            <Button variant="outline" size="sm">
                              <Eye className="h-4 w-4 mr-1" />
                              Review
                            </Button>
                            <Button variant="outline" size="sm">
                              <Play className="h-4 w-4 mr-1" />
                              Play Video
                            </Button>
                            <Button variant="outline" size="sm">
                              <Download className="h-4 w-4 mr-1" />
                              Download
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="review" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Manual Review Interface</CardTitle>
                <CardDescription>Detailed review interface for flagged KYC sessions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <MessageSquare className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Select a flagged session to begin manual review</p>
                  <p className="text-sm text-gray-400 mt-2">
                    Video player, document viewer, and decision interface would be displayed here
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="audit" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Audit Log</CardTitle>
                <CardDescription>Complete audit trail of all KYC verification activities</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Clock className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Audit log entries would be displayed here</p>
                  <p className="text-sm text-gray-400 mt-2">
                    Timestamps, actions, and user activities for compliance tracking
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
