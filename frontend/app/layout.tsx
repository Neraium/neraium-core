import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Neraium System Intelligence',
  description: 'Real-time system intelligence and gate decision replay',
  icons: {
    icon: '/neraium-logo.jpg',
    shortcut: '/neraium-logo.jpg',
    apple: '/neraium-logo.jpg',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
