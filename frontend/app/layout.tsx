import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Neraium System Intelligence',
  description: 'Real-time system intelligence and gate decision replay',
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
