'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface NarrativeLayerProps {
  text: string
}

export function NarrativeLayer({ text }: NarrativeLayerProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 1.2, ease: 'easeInOut' }}
      style={{
        textAlign: 'center',
        maxWidth: '600px',
        margin: '0 auto',
      }}
    >
      <p
        style={{
          fontSize: '16px',
          lineHeight: '1.6',
          color: '#cbd5e1',
          fontWeight: 400,
          margin: 0,
          letterSpacing: '0.3px',
        }}
      >
        {text}
      </p>
    </motion.div>
  )
}
