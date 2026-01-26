import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

// DO NOT import index.css here if you deleted it
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)