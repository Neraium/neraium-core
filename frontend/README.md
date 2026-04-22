# Neraium System Intelligence UI

Next.js frontend for the Neraium system intelligence demo.

## Setup

```bash
npm install
```

## Development

```bash
# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

The app expects the FastAPI backend to be running on `http://localhost:8000`.

Set `NEXT_PUBLIC_API_URL` in `.env.local` to change the API endpoint:

```
NEXT_PUBLIC_API_URL=http://your-api-host:8000
```

## Build

```bash
npm run build
npm start
```

## Features

- Frame-by-frame replay with interactive slider
- Playback controls (play, pause, speed adjustment)
- Real-time system state visualization
- Tetrahedron structural state projection
- Comprehensive insights and metrics
- Responsive design (desktop-first)
- Synthetic fallback demo mode if API is unavailable
