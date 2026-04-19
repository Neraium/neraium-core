/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_NERAIUM_API_BASE || 'http://localhost:8000'}/api/:path*`,
        },
      ],
    }
  },
}

module.exports = nextConfig
