'use client'

import dynamic from 'next/dynamic'
import { Suspense } from 'react'

const LouddpaxRoomIntelligence = dynamic(
  () => import('@/components/LouddpaxRoomIntelligence').then(mod => ({ default: mod.LouddpaxRoomIntelligence })),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-screen bg-louddpax-bg">
        <div className="text-louddpax-text">Loading...</div>
      </div>
    )
  }
)

export default function Home() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen bg-louddpax-bg"><div className="text-louddpax-text">Loading...</div></div>}>
      <LouddpaxRoomIntelligence />
    </Suspense>
  )
}
