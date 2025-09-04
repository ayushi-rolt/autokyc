"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { CheckCircle, Play, User, FileText, Shield } from 'lucide-react'

const demoSteps = [
  {
    id: "upload",
    title: "Document uploaded successfully",
    icon: FileText,
    status: "completed"
  },
  {
    id: "liveness", 
    title: "Liveness check passed",
    icon: User,
    status: "completed"
  },
  {
    id: "verification",
    title: "Identity Verified",
    icon: Shield,
    status: "completed",
    badge: "Risk: Low"
  }
]

export function InteractiveDemo() {
  const [isPlaying, setIsPlaying] = useState(false)

  const startDemo = () => {
    setIsPlaying(true)
    // Reset after demo
    setTimeout(() => setIsPlaying(false), 5000)
  }

  return (
    <section className="mb-16">
      <div className="grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-6">
            See SecureKYC in Action
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            Experience our verification process firsthand. Test with sample documents and see how 
            quickly and accurately SecureKYC validates identities while maintaining the highest security standards.
          </p>
          
          <div className="space-y-4 mb-8">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span>No registration required</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span>Test with sample documents</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span>See real-time results</span>
            </div>
          </div>

          <div className="flex space-x-4">
            <Button size="lg" onClick={startDemo}>
              Start Demo
              <Play className="h-5 w-5 ml-2" />
            </Button>
            <Button variant="outline" size="lg">
              Watch Video
            </Button>
          </div>
        </div>

        <div>
          <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-800 dark:to-gray-700 border-0">
            <CardContent className="p-8">
              <div className="bg-gray-900 rounded-lg p-4 mb-6">
                <div className="flex items-center space-x-2 mb-4">
                  <User className="h-5 w-5 text-blue-400" />
                  <span className="text-white font-medium">Try Interactive Demo</span>
                </div>
              </div>

              <div className="space-y-4">
                {demoSteps.map((step, index) => {
                  const IconComponent = step.icon
                  return (
                    <div 
                      key={step.id}
                      className={`flex items-center justify-between p-4 rounded-lg transition-all ${
                        isPlaying 
                          ? 'bg-white dark:bg-gray-800 shadow-sm' 
                          : 'bg-gray-100 dark:bg-gray-700'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-full ${
                          step.status === 'completed' ? 'bg-green-100 dark:bg-green-900/30' : 'bg-gray-200 dark:bg-gray-600'
                        }`}>
                          <IconComponent className={`h-4 w-4 ${
                            step.status === 'completed' ? 'text-green-600' : 'text-gray-500'
                          }`} />
                        </div>
                        <span className="font-medium">{step.title}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {step.badge && (
                          <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 text-xs rounded-full">
                            {step.badge}
                          </span>
                        )}
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}