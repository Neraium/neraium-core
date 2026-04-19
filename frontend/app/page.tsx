import dynamic from 'next/dynamic'

const LouddpaxRoomIntelligence = dynamic(
  () => import('@/components/LouddpaxRoomIntelligence').then(mod => ({ default: mod.LouddpaxRoomIntelligence })),
  { ssr: false, loading: () => <div className="flex items-center justify-center h-screen">Loading...</div> }
)

export default function Home() {
  return <LouddpaxRoomIntelligence />
}
