import { useCallback, useState } from 'react';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatInputProps {
  onSendMessage: (text: string) => void;
  loading?: boolean;
}

// ---------------------------------------------------------------------------
// Suggestion chips
// ---------------------------------------------------------------------------

const SUGGESTIONS = [
  'Why not another country?',
  'Try searching nearby landmarks',
  'Compare candidates',
  'Show evidence breakdown',
  'What clues are in the image?',
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function ChatInput({ onSendMessage, loading = false }: ChatInputProps) {
  const [value, setValue] = useState('');

  const handleSubmit = useCallback(
    (text?: string) => {
      const trimmed = (text ?? value).trim();
      if (!trimmed || loading) return;
      onSendMessage(trimmed);
      setValue('');
    },
    [value, loading, onSendMessage],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <div className="border-t border-gray-200 bg-white px-4 pb-4 pt-3">
      {/* Suggestion chips */}
      <div className="mb-2 flex flex-wrap gap-1.5">
        {SUGGESTIONS.map((chip) => (
          <button
            key={chip}
            type="button"
            disabled={loading}
            onClick={() => handleSubmit(chip)}
            className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {chip}
          </button>
        ))}
      </div>

      {/* Input row */}
      <div className="flex items-end gap-2">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={loading ? 'Waiting for response...' : 'Ask a follow-up question...'}
          disabled={loading}
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-2.5 text-sm text-gray-800 placeholder-gray-400 outline-none transition-colors focus:border-blue-500 focus:bg-white focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        />

        <button
          type="button"
          onClick={() => handleSubmit()}
          disabled={loading || !value.trim()}
          className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-400"
          aria-label="Send message"
        >
          {loading ? (
            <div className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
          ) : (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 12h14M12 5l7 7-7 7"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}

export default ChatInput;
