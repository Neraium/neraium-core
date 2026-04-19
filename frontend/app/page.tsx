import dynamic from 'next/dynamic'

const LouddpaxRoomIntelligence = dynamic(
  () => import('@/components/LouddpaxRoomIntelligence').then(mod => ({ default: mod.LouddpaxRoomIntelligence })),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-screen bg-louddpax-bg">
        <div className="animate-pulse text-louddpax-text">Loading Intelligence...</div>
      </div>
    )
  }
)

export default function Home() {
  return (
    <div suppressHydrationWarning>
      <LouddpaxRoomIntelligence />
    </div>
  )
}
