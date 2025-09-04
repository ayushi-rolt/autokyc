import Link from "next/link"
import { Fingerprint, Mail, Phone, MapPin } from 'lucide-react'

export function Footer() {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-16">
        <div className="grid md:grid-cols-5 gap-8">
          {/* Company Info */}
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <Fingerprint className="h-8 w-8 text-blue-400" />
              <span className="text-xl font-bold">SecureKYC</span>
            </div>
            <p className="text-gray-400 mb-6">
              The most trusted identity verification platform for financial institutions worldwide.
            </p>
            <div className="space-y-2 text-sm text-gray-400">
              <div className="flex items-center space-x-2">
                <Mail className="h-4 w-4" />
                <span>hello@securekyc.com</span>
              </div>
              <div className="flex items-center space-x-2">
                <Phone className="h-4 w-4" />
                <span>+1 (555) 123-4567</span>
              </div>
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4" />
                <span>London, UK</span>
              </div>
            </div>
          </div>

          {/* Product */}
          <div>
            <h3 className="font-semibold mb-4">Product</h3>
            <div className="space-y-2 text-sm text-gray-400">
              <Link href="#" className="block hover:text-white">Document Verification</Link>
              <Link href="#" className="block hover:text-white">Facial Biometrics</Link>
              <Link href="#" className="block hover:text-white">AML Screening</Link>
              <Link href="#" className="block hover:text-white">API Documentation</Link>
              <Link href="#" className="block hover:text-white">Pricing</Link>
            </div>
          </div>

          {/* Solutions */}
          <div>
            <h3 className="font-semibold mb-4">Solutions</h3>
            <div className="space-y-2 text-sm text-gray-400">
              <Link href="#" className="block hover:text-white">Financial Services</Link>
              <Link href="#" className="block hover:text-white">Fintech</Link>
              <Link href="#" className="block hover:text-white">Cryptocurrency</Link>
              <Link href="#" className="block hover:text-white">Gaming</Link>
              <Link href="#" className="block hover:text-white">E-commerce</Link>
            </div>
          </div>

          {/* Compliance */}
          <div>
            <h3 className="font-semibold mb-4">Compliance</h3>
            <div className="space-y-2 text-sm text-gray-400">
              <Link href="#" className="block hover:text-white">GDPR</Link>
              <Link href="#" className="block hover:text-white">AML/CTF</Link>
              <Link href="#" className="block hover:text-white">PCI DSS</Link>
              <Link href="#" className="block hover:text-white">ISO 27001</Link>
              <Link href="#" className="block hover:text-white">Certifications</Link>
            </div>
          </div>

          {/* Support */}
          <div>
            <h3 className="font-semibold mb-4">Support</h3>
            <div className="space-y-2 text-sm text-gray-400">
              <Link href="#" className="block hover:text-white">Help Center</Link>
              <Link href="#" className="block hover:text-white">API Docs</Link>
              <Link href="#" className="block hover:text-white">Status Page</Link>
              <Link href="#" className="block hover:text-white">Contact Support</Link>
              <Link href="#" className="block hover:text-white">Partner Portal</Link>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-12 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm text-gray-400 mb-4 md:mb-0">
              © 2024 SecureKYC. All rights reserved.
            </div>
            <div className="flex space-x-6 text-sm text-gray-400">
              <Link href="#" className="hover:text-white">Privacy Policy</Link>
              <Link href="#" className="hover:text-white">Terms of Service</Link>
              <Link href="#" className="hover:text-white">Cookie Policy</Link>
              <div className="flex space-x-4">
                <span>GDPR Compliant</span>
                <span>ISO 27001</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}