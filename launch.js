#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const projectRoot = __dirname;

console.log('🚀 Starting NeRAIUM...\n');

// Start backend
const backend = spawn('python', [
  '-m', 'uvicorn', 'apps.api.main:app',
  '--reload', '--port', '8000'
], {
  cwd: projectRoot,
  stdio: 'inherit'
});

// Start frontend after 3 seconds
setTimeout(() => {
  const frontend = spawn('npm', ['run', 'dev'], {
    cwd: path.join(projectRoot, 'frontend'),
    stdio: 'inherit'
  });
}, 3000);

console.log('\n✓ Backend: http://localhost:8000');
console.log('✓ Frontend: http://localhost:3000\n');

process.on('SIGINT', () => {
  console.log('\n\nShutting down...');
  process.exit(0);
});
