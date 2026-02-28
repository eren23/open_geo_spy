import { useCallback, useState } from 'react';
import type { ChatMessage } from '../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UseChatMessagesReturn {
  /** Current list of chat messages. */
  messages: ChatMessage[];
  /** Append a fully-formed message. */
  addMessage: (msg: ChatMessage) => void;
  /** Convenience: append a pipeline agent step as a system message. */
  addAgentStep: (step: string, status: string, detail?: string) => void;
  /** Remove all messages. */
  clear: () => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Simple state container for a chat message list.
 *
 * Keeps an ordered array of `ChatMessage` objects and exposes helpers for
 * the most common mutations: adding a raw message, appending a pipeline
 * step summary (rendered as a system message), and clearing the history.
 */
export function useChatMessages(
  initial: ChatMessage[] = [],
): UseChatMessagesReturn {
  const [messages, setMessages] = useState<ChatMessage[]>(initial);

  const addMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev: ChatMessage[]) => [...prev, msg]);
  }, []);

  const addAgentStep = useCallback(
    (step: string, status: string, detail?: string) => {
      const content = detail
        ? `[${step}] ${status}: ${detail}`
        : `[${step}] ${status}`;

      const msg: ChatMessage = {
        role: 'system',
        content,
        timestamp: new Date().toISOString(),
        metadata: { step, status, detail },
      };

      setMessages((prev: ChatMessage[]) => {
        // Find existing message for this step and replace it
        const idx = prev.findIndex(
          (m) => m.role === 'system' && m.metadata?.step === step,
        );
        if (idx !== -1) {
          const updated = [...prev];
          updated[idx] = msg;
          return updated;
        }
        return [...prev, msg];
      });
    },
    [],
  );

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, addMessage, addAgentStep, clear };
}

export default useChatMessages;
