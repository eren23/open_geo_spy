import type { ChatMessage } from '../../types';
import MessageList from './MessageList';
import ChatInput from './ChatInput';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatContainerProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  loading?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Top-level chat panel that composes MessageList and ChatInput.
 *
 * Expects its parent to control width/height -- the container fills the
 * available space via `flex flex-col h-full`.
 */
function ChatContainer({ messages, onSendMessage, loading = false }: ChatContainerProps) {
  return (
    <div className="flex h-full flex-col rounded-xl border border-gray-200 bg-white shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-100 text-blue-600">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
            />
          </svg>
        </div>
        <h2 className="text-sm font-semibold text-gray-700">Chat</h2>
        {loading && (
          <span className="ml-auto text-xs text-gray-400">Thinking...</span>
        )}
      </div>

      {/* Message list (scrollable) */}
      <MessageList messages={messages} />

      {/* Input area */}
      <ChatInput onSendMessage={onSendMessage} loading={loading} />
    </div>
  );
}

export default ChatContainer;
