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
    <div className="flex h-full flex-col bg-white">
      {/* Message list (scrollable) */}
      <MessageList messages={messages} />

      {/* Input area */}
      <ChatInput onSendMessage={onSendMessage} loading={loading} />
    </div>
  );
}

export default ChatContainer;
