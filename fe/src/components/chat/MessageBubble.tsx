import type { ChatMessage } from '../../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MessageBubbleProps {
  message: ChatMessage;
}

// ---------------------------------------------------------------------------
// Helpers -- lightweight markdown-ish formatting
// ---------------------------------------------------------------------------

/**
 * Very small inline formatter.  Handles:
 *  - **bold**
 *  - `inline code`
 *  - ```code blocks```
 *
 * Returns an array of React nodes so we stay within plain React (no deps).
 */
function formatContent(raw: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let key = 0;

  // Split on fenced code blocks first (``` ... ```)
  const codeBlockParts = raw.split(/```([\s\S]*?)```/);

  for (let i = 0; i < codeBlockParts.length; i++) {
    if (i % 2 === 1) {
      // Inside a fenced code block
      nodes.push(
        <pre
          key={key++}
          className="my-2 rounded-md bg-gray-900 p-3 text-xs text-gray-100 overflow-x-auto"
        >
          <code>{codeBlockParts[i].trim()}</code>
        </pre>,
      );
    } else {
      // Normal text -- handle inline patterns
      const text = codeBlockParts[i];
      // Split on inline code and bold
      const inlineParts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/);

      for (const part of inlineParts) {
        if (part.startsWith('**') && part.endsWith('**')) {
          nodes.push(
            <strong key={key++} className="font-semibold">
              {part.slice(2, -2)}
            </strong>,
          );
        } else if (part.startsWith('`') && part.endsWith('`')) {
          nodes.push(
            <code
              key={key++}
              className="rounded bg-gray-200 px-1 py-0.5 text-xs font-mono text-gray-800"
            >
              {part.slice(1, -1)}
            </code>,
          );
        } else if (part) {
          nodes.push(<span key={key++}>{part}</span>);
        }
      }
    }
  }

  return nodes;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : 'bg-gray-100 text-gray-800 rounded-bl-md'
        }`}
      >
        <div className="whitespace-pre-wrap break-words">
          {formatContent(message.content)}
        </div>

        {message.timestamp && (
          <div
            className={`mt-1 text-[10px] ${
              isUser ? 'text-blue-200' : 'text-gray-400'
            }`}
          >
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default MessageBubble;
