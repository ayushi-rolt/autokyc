import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FileText, Scan, Shield, CheckCircle, ArrowRight } from 'lucide-react'

const suiteItems = [
  {
    title: "Document Verification",
    description: "AI-powered validation of 3,000+ document types from 195+ countries",
    features: [
      "OCR Text Extraction",
      "Fraud Detection", 
      "Real-time Validation"
    ],
    icon: FileText,
    color: "bg-blue-600",
    image: "/placeholder.svg?height=200&width=300&text=Document+Verification"
  },
  {
    title: "Facial Biometrics",
    description: "Advanced liveness detection and face matching technology",
    features: [
      "Liveness Detection",
      "Face Matching",
      "Anti-Spoofing"
    ],
    icon: Scan,
    color: "bg-green-600", 
    image: "/placeholder.svg?height=200&width=300&text=Facial+Biometrics"
  },
  {
    title: "Compliance Dashboard",
    description: "Comprehensive AML, PEP, and sanctions screening with risk assessment",
    features: [
      "PEP Screening",
      "Sanctions Check",
      "Risk Scoring"
    ],
    icon: Shield,
    color: "bg-purple-600",
    image: "/placeholder.svg?height=200&width=300&text=Compliance+Dashboard"
  }
]

export function IdentityVerificationSuite() {
  return (
    <section className="mb-16" id="solutions">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Complete <span className="text-blue-600">Identity Verification</span> Suite
        </h2>
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          Everything you need to verify customer identities and ensure compliance with global 
          KYC and AML regulations.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {suiteItems.map((item, index) => {
          const IconComponent = item.icon
          return (
            <Card key={index} className="overflow-hidden hover:shadow-xl transition-shadow">
              <div className="relative h-48 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800">
                <img 
                  src={item.image || "/placeholder.svg"} 
                  alt={item.title}
                  className="w-full h-full object-cover"
                />
                <div className={`absolute top-4 left-4 ${item.color} p-2 rounded-lg`}>
                  <IconComponent className="h-6 w-6 text-white" />
                </div>
              </div>
              
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  {item.title}
                </CardTitle>
              </CardHeader>
              
              <CardContent>
                <p className="text-gray-600 dark:text-gray-400 mb-4">{item.description}</p>
                
                <div className="space-y-2 mb-6">
                  {item.features.map((feature, featureIndex) => (
                    <div key={featureIndex} className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-600" />
                      <span className="text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
                
                <Button variant="outline" className="w-full">
                  Learn More
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </section>
  )
}