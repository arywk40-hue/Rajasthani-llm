import React from 'react';
import { ArrowRight, Loader2 } from 'lucide-react';

export function TranslateButton({ onClick, isLoading, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`px-8 py-3.5 rounded-full font-medium text-sm transition-all duration-300 flex items-center gap-2.5 shadow-lg ${
        disabled || isLoading
          ? 'bg-[#1D1D24] text-gray-500 cursor-not-allowed border border-white/5'
          : 'bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white shadow-purple-900/30 hover:shadow-purple-900/50 hover:scale-[1.02] active:scale-[0.98]'
      }`}
    >
      {isLoading ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin text-purple-200" />
          <span>Translating...</span>
        </>
      ) : (
        <>
          <span>Translate</span>
          <ArrowRight className="w-4 h-4" />
        </>
      )}
    </button>
  );
}
