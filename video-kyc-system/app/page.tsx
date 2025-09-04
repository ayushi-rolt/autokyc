"use client";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock, Users, FileText, CheckCircle, Video, Shield, ArrowRight, Play } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { Fingerprint } from "lucide-react";
import { FeaturesCarousel } from "@/components/features-carousel";
import { ProcessSteps } from "@/components/process-steps";
import { IdentityVerificationSuite } from "@/components/identity-verification-suite";
import { InteractiveDemo } from "@/components/interactive-demo";
import { TrustedBySection } from "@/components/trusted-by-section";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Fingerprint className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">SecureKYC</span>
          </div>
          <nav className="hidden md:flex items-center space-x-6">
            <Link href="#features" className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">Features</Link>
            <Link href="#solutions" className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">Solutions</Link>
            <Link href="#compliance" className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">Compliance</Link>
            <Link href="#pricing" className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">Pricing</Link>
            <Link href="/auth/login" className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">Sign In</Link>
            <ThemeToggle />
            <Link href="/kyc/start"><Button>Start Verification</Button></Link>
          </nav>
        </div>
      </header>
      {/* Hero Section */}
      <main className="container mx-auto px-4 py-16">
        <section className="text-center mb-16 max-w-4xl mx-auto">
          {/* info pill */}
          <div className="inline-flex items-center px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-full text-sm text-blue-800 dark:text-blue-200 mb-6">
            <Shield className="h-4 w-4 mr-2" />
            EU-Compliant Identity Verification
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
            Automated KYC Verification in <span className="text-blue-600">Real-Time</span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
            Streamline your customer onboarding with AI-powered identity verification, document validation, and compliance screening. Reduce fraud while enhancing user experience.
          </p>
          {/* Key Stats Card Usage */}
          <div className="flex flex-wrap justify-center gap-8 mb-8">
            {[{
              icon: <CheckCircle className="h-5 w-5 text-green-600" />,
              label: "99.8% Accuracy"
            }, {
              icon: <Clock className="h-5 w-5 text-blue-600" />,
              label: "30s Verification"
            }, {
              icon: <Shield className="h-5 w-5 text-purple-600" />,
              label: "AML Compliant"
            }].map((item, idx) => (
              <Card key={idx} className="min-w-[140px] flex flex-col items-center shadow-md">
                <CardHeader>
                  <CardTitle>{item.icon}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>{item.label}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/kyc/start">
              <Button size="lg" className="text-lg px-8 py-4">
                Start Free Trial <ArrowRight className="h-5 w-5 ml-2" />
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="text-lg px-8 py-4">
              <Play className="h-5 w-5 mr-2" /> Watch Demo
            </Button>
          </div>
        </section>
        {/* Demo Card Example */}
        <section className="relative max-w-4xl mx-auto mb-16">
          <Card className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8 border">
            <CardHeader>
              <CardTitle>Verification Complete</CardTitle>
              <CardDescription>securekyc.com/verify</CardDescription>
            </CardHeader>
            <CardContent className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg p-8 text-white text-center">
              <div className="inline-flex items-center px-3 py-1 bg-green-500 rounded-full text-sm mb-4"><CheckCircle className="h-4 w-4 mr-1" /> Identity Verified in 28 seconds</div>
              <p className="opacity-90">Customer successfully onboarded with full compliance</p>
            </CardContent>
          </Card>
        </section>
        {/* Trusted By Section */}
        <TrustedBySection />
        {/* Process Steps */}
        <ProcessSteps />
        {/* Identity Verification Suite */}
        <IdentityVerificationSuite />
        {/* Features Carousel */}
        <FeaturesCarousel />
        {/* Interactive Demo */}
        <InteractiveDemo />
        {/* Stats Section with Card Group */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-8 mb-16">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            {[{
              icon: <Clock className="h-6 w-6 text-blue-600 mr-2" />,
              stat: "2 min",
              desc: "Average completion time"
            }, {
              icon: <Shield className="h-6 w-6 text-green-600 mr-2" />,
              stat: "99.8%",
              desc: "Accuracy rate"
            }, {
              icon: <Users className="h-6 w-6 text-purple-600 mr-2" />,
              stat: "50K+",
              desc: "Verifications completed"
            }].map((item, idx) => (
              <Card key={idx} className="flex flex-col items-center shadow-sm">
                <CardHeader>
                  <CardTitle>{item.stat}</CardTitle>
                </CardHeader>
                <CardContent>
                  {item.icon}
                  <CardDescription>{item.desc}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
