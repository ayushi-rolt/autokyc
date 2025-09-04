"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Video, FileText, CheckCircle, Shield, Zap, Globe } from 'lucide-react'

const features = [
  {
    icon: Video,
    title: "Video Verification",
    description: "Real-time liveness detection and face matching using advanced AI algorithms",
    color: "text-blue-600",
    bgColor: "bg-blue-50 dark:bg-blue-900/20"
  },
  {
    icon: FileText,
    title: "Document OCR",
    description: "Automatic extraction and verification of Aadhaar and PAN card details",
    color: "text-green-600",
    bgColor: "bg-green-50 dark:bg-green-900/20"
  },
  {
    icon: CheckCircle,
    title: "Instant Results",
    description: "Get verification results within minutes with detailed audit trails",
    color: "text-purple-600",
    bgColor: "bg-purple-50 dark:bg-purple-900/20"
  },
  {
    icon: Shield,
    title: "Fraud Detection",
    description: "Advanced AI-powered fraud detection and risk assessment",
    color: "text-red-600",
    bgColor: "bg-red-50 dark:bg-red-900/20"
  },
  {
    icon: Zap,
    title: "Real-time Processing",
    description: "Complete verification in under 30 seconds with instant results",
    color: "text-yellow-600",
    bgColor: "bg-yellow-50 dark:bg-yellow-900/20"
  },
  {
    icon: Globe,
    title: "Global Coverage",
    description: "Support for 195+ countries and 3,000+ document types",
    color: "text-indigo-600",
    bgColor: "bg-indigo-50 dark:bg-indigo-900/20"
  }
]

export function FeaturesCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(true)

  useEffect(() => {
    if (!isAutoPlaying) return

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % features.length)
    }, 4000)

    return () => clearInterval(interval)
  }, [isAutoPlaying])

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev + 1) % features.length)
    setIsAutoPlaying(false)
  }

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev - 1 + features.length) % features.length)
    setIsAutoPlaying(false)
  }

  const goToSlide = (index: number) => {
    setCurrentIndex(index)
    setIsAutoPlaying(false)
  }

  return (
    <section className="mb-16">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Powerful Features
        </h2>
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Everything you need for comprehensive identity verification and compliance
        </p>
      </div>

      <div className="relative max-w-4xl mx-auto">
        {/* Carousel Container */}
        <div className="overflow-hidden rounded-2xl">
          <div 
            className="flex transition-transform duration-500 ease-in-out"
            style={{ transform: `translateX(-${currentIndex * 100}%)` }}
          >
            {features.map((feature, index) => {
              const IconComponent = feature.icon
              return (
                <div key={index} className="w-full flex-shrink-0">
                  <Card className={`${feature.bgColor} border-0 h-80`}>
                    <CardContent className="flex flex-col items-center justify-center h-full text-center p-8">
                      <div className={`${feature.bgColor} p-4 rounded-full mb-6`}>
                        <IconComponent className={`h-12 w-12 ${feature.color}`} />
                      </div>
                      <CardTitle className="text-2xl mb-4">{feature.title}</CardTitle>
                      <CardDescription className="text-lg max-w-md">
                        {feature.description}
                      </CardDescription>
                    </CardContent>
                  </Card>
                </div>
              )
            })}
          </div>
        </div>

        {/* Navigation Buttons */}
        <Button
          variant="outline"
          size="icon"
          className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm"
          onClick={prevSlide}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm"
          onClick={nextSlide}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        {/* Dots Indicator */}
        <div className="flex justify-center space-x-2 mt-6">
          {features.map((_, index) => (
            <button
              key={index}
              className={`w-3 h-3 rounded-full transition-colors ${
                index === currentIndex 
                  ? 'bg-blue-600' 
                  : 'bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500'
              }`}
              onClick={() => goToSlide(index)}
            />
          ))}
        </div>
      </div>
    </section>
  )
}