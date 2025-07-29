import Link from "next/link"
import { Button } from "@/components/ui/button"
import { FileText, Home, BarChart3, Camera } from "lucide-react"

export function Navbar() {
  return (
    <nav className="border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <FileText className="w-6 h-6 text-blue-600" />
            <span className="text-xl font-bold">AutoKYC</span>
          </Link>

          <div className="flex items-center space-x-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <Home className="w-4 h-4 mr-2" />
                Home
              </Button>
            </Link>
            <Link href="/capture">
              <Button variant="ghost" size="sm">
                <Camera className="w-4 h-4 mr-2" />
                Capture
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">
                <BarChart3 className="w-4 h-4 mr-2" />
                Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
