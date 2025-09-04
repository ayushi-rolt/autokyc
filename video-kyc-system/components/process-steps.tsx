import { Card, CardContent } from "@/components/ui/card"
import { Upload, Camera, Shield, CheckCircle, Clock } from 'lucide-react'

const steps = [
  {
    number: "01",
    icon: Upload,
    title: "Upload Document",
    description: "Upload ID, passport, or driver's license",
    time: "5 seconds",
    color: "bg-blue-600"
  },
  {
    number: "02",
    icon: Camera,
    title: "Take Selfie",
    description: "Capture live selfie with liveness detection",
    time: "10 seconds",
    color: "bg-green-600"
  },
  {
    number: "03",
    icon: Shield,
    title: "AI Verification",
    description: "Advanced AI validates and cross-references data",
    time: "15 seconds",
    color: "bg-purple-600"
  },
  {
    number: "04",
    icon: CheckCircle,
    title: "Instant Results",
    description: "Get verification status and compliance report",
    time: "Instant",
    color: "bg-orange-600"
  }
]

export function ProcessSteps() {
  return (
    <section className="mb-16" id="features">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Simple <span className="text-green-600">4-Step Process</span>
        </h2>
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Complete identity verification in under 30 seconds with our streamlined workflow
        </p>
      </div>

      <div className="grid md:grid-cols-4 gap-6">
        {steps.map((step, index) => {
          const IconComponent = step.icon
          return (
            <div key={index} className="relative">
              <Card className="h-full hover:shadow-lg transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="relative mb-6">
                    <div className={`w-12 h-12 ${step.color} text-white rounded-full flex items-center justify-center text-lg font-bold mx-auto mb-4`}>
                      {step.number}
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-full w-fit mx-auto">
                      <IconComponent className="h-8 w-8 text-gray-600 dark:text-gray-300" />
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                  <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">{step.description}</p>
                  <div className="flex items-center justify-center space-x-1 text-sm text-gray-500">
                    <Clock className="h-4 w-4" />
                    <span>{step.time}</span>
                  </div>
                </CardContent>
              </Card>
              
              {/* Arrow connector */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                  <div className="w-6 h-0.5 bg-gray-300 dark:bg-gray-600"></div>
                  <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-0 h-0 border-l-4 border-l-gray-300 dark:border-l-gray-600 border-t-2 border-b-2 border-t-transparent border-b-transparent"></div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}