import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// No StrictMode: its double-invoked effects would revoke the preview blob URL
// while the <img> still points at it.
createRoot(document.getElementById('root')!).render(<App />)
