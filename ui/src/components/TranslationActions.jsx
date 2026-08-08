import React, { useState } from 'react';
import { Copy, Check, RefreshCw, X } from 'lucide-react';

export function TranslationActions({ text, onClear, onRegenerate, isInput = false }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!text) return null;

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={handleCopy}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white hover:bg-[#1D1D24] transition-colors"
        title="Copy text"
      >
        {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
        <span>{copied ? 'Copied' : 'Copy'}</span>
      </button>

      {isInput ? (
        <button
          onClick={onClear}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white hover:bg-[#1D1D24] transition-colors"
          title="Clear text"
        >
          <X className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      ) : (
        <button
          onClick={onRegenerate}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white hover:bg-[#1D1D24] transition-colors"
          title="Regenerate translation"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Regenerate</span>
        </button>
      )}
    </div>
  );
}
