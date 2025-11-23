import { useState, useEffect, useRef } from 'react';

export const useWebSocket = (url, enabled = true) => {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!enabled || !url) return;

    const wsUrl = url.replace('http', 'ws');
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected:', wsUrl);
      setStatus('connected');
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setData(message);
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setError('Connection error');
      setStatus('error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus('disconnected');
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [url, enabled]);

  return { data, status, error };
};
