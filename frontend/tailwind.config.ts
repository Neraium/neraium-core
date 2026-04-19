import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Louddpax brand colors
        louddpax: {
          primary: '#7E9F2E',      // System green
          drift: '#D8A35D',         // Amber drift indicator
          critical: '#C94C4C',      // Critical red
          bg: '#050607',            // Deep black background
          surface: '#0B0D0F',       // Card/surface background
          border: '#1A1D22',        // Border color
          text: '#E8EBF0',          // Primary text
          muted: '#8B92A0',         // Muted text
          accent: '#A3E635',        // Secondary accent (lime)
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          '"Helvetica Neue"',
          'sans-serif',
        ],
        mono: ['Menlo', 'Monaco', 'monospace'],
      },
      fontSize: {
        xs: ['0.65rem', { lineHeight: '0.85rem', letterSpacing: '0.08em' }],
        sm: ['0.75rem', { lineHeight: '1rem', letterSpacing: '0.06em' }],
        base: ['0.875rem', { lineHeight: '1.25rem', letterSpacing: '0.04em' }],
        lg: ['1rem', { lineHeight: '1.5rem', letterSpacing: '0.02em' }],
        xl: ['1.25rem', { lineHeight: '1.75rem', letterSpacing: '0em' }],
        '2xl': ['1.5rem', { lineHeight: '2rem', letterSpacing: '-0.01em' }],
        '3xl': ['2rem', { lineHeight: '2.5rem', letterSpacing: '-0.02em' }],
      },
      boxShadow: {
        subtle: '0 4px 12px rgba(0, 0, 0, 0.3)',
        base: '0 8px 24px rgba(0, 0, 0, 0.4)',
        lg: '0 16px 48px rgba(0, 0, 0, 0.5)',
        glow: '0 0 20px rgba(126, 159, 46, 0.2)',
        'glow-critical': '0 0 24px rgba(201, 76, 76, 0.25)',
      },
      animation: {
        breathe: 'breathe 4s ease-in-out infinite',
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        drift: 'drift 3s ease-in-out infinite',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.05)', opacity: '0.8' },
        },
        drift: {
          '0%, 100%': { transform: 'translateX(0)', opacity: '0.5' },
          '50%': { transform: 'translateX(2px)', opacity: '0.7' },
        },
      },
      transitionTimingFunction: {
        smooth: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
        bounce: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
    },
  },
  plugins: [],
}
export default config
