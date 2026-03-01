import { useEffect, useRef } from 'react';
import type { ChatMessage } from '../../types';
import MessageBubble from './MessageBubble';
import AgentStepMessage from './AgentStepMessage';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MessageListProps {
  messages: ChatMessage[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
        Waiting for pipeline events...
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.map((msg, idx) => {
        // System messages created by addAgentStep carry step metadata
        const isAgentStep =
          msg.role === 'system' && msg.metadata?.step != null;

        if (isAgentStep) {
          return <AgentStepMessage key={idx} message={msg} />;
        }

        // Skip bare system messages that have no step metadata
        if (msg.role === 'system') {
          return null;
        }

        return <MessageBubble key={idx} message={msg} />;
      })}

      {/* Invisible scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}

export default MessageList;
