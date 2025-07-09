import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/globals.css';
import App from './App.tsx';
import { Toaster } from 'react-hot-toast';

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <App />
        <Toaster
            toastOptions={{
                style: {
                    backgroundColor: '#171717',
                    color: 'white'
                }
            }}
        />
    </StrictMode>
);
