import { useEffect, useRef, useCallback } from "react";
import { useSimStore } from "../state/store";
import type { ControlMessage, SimMessage } from "../lib/types";

const WS_URL = "ws://localhost:8000/ws/sim";
const RECONNECT_DELAY_MS = 2000;

export function useSimSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { setLatestState, setLatestStats, addGenerationRecord, setConnected } =
    useSimStore();

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    const socket = new WebSocket(WS_URL);
    ws.current = socket;

    socket.onopen = () => {
      setConnected(true);
    };

    socket.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    socket.onerror = () => {
      socket.close();
    };

    socket.onmessage = (ev: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(ev.data) as SimMessage;
        if (msg.type === "state") {
          setLatestState(msg);
        } else if (msg.type === "stats") {
          setLatestStats(msg);
        } else if (msg.type === "generation") {
          addGenerationRecord(msg);
        }
      } catch {
        // ignore malformed frames
      }
    };
  }, [setLatestState, setLatestStats, addGenerationRecord, setConnected]);

  useEffect(() => {
    connect();
    return () => {
      reconnectTimer.current && clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: ControlMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  }, []);

  return { send };
}
