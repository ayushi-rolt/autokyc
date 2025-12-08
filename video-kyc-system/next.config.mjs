/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // 👇 ADD THIS SECTION TO CONNECT TO FASTAPI 👇
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // This directs requests to your FastAPI backend
        destination: 'http://127.0.0.1:8000/api/:path*', 
      },
    ]
  },
}

export default nextConfig