import React from 'react';
import { LanguageSelector } from './LanguageSelector';
import { TranslationActions } from './TranslationActions';

export function TranslationBox({
  label,
  language,
  onLanguageChange,
  text,
  onTextChange,
  isReadOnly = false,
  placeholder,
  onClear,
  onRegenerate,
  maxCharCount = 5000
}) {
  const charCount = text ? text.length : 0;

  return (
    <div className="flex-1 bg-[#15151A] border border-white/10 rounded-2xl p-5 flex flex-col justify-between min-h-[320px] transition-all hover:border-white/20 shadow-xl">
      {/* Header Bar */}
      <div className="flex items-center justify-between pb-4 border-b border-white/5">
        <LanguageSelector value={language} onChange={onLanguageChange} />
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</span>
      </div>

      {/* Content Area */}
      <div className="flex-1 my-4">
        {isReadOnly ? (
          <div className="w-full h-full text-base text-gray-200 leading-relaxed font-normal whitespace-pre-wrap select-text">
            {text || <span className="text-gray-600 italic">{placeholder}</span>}
          </div>
        ) : (
          <textarea
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder={placeholder}
            maxLength={maxCharCount}
            className="w-full h-full bg-transparent text-base text-white placeholder-gray-600 outline-none resize-none leading-relaxed font-normal"
          />
        )}
      </div>

      {/* Footer Bar */}
      <div className="flex items-center justify-between pt-3 border-t border-white/5">
        <TranslationActions
          text={text}
          onClear={onClear}
          onRegenerate={onRegenerate}
          isInput={!isReadOnly}
        />

        {!isReadOnly && (
          <span className="text-xs font-mono text-gray-400">
            {charCount} / {maxCharCount}
          </span>
        )}
      </div>
    </div>
  );
}
