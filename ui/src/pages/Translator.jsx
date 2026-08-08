import React from 'react';
import { ArrowLeftRight, Sparkles } from 'lucide-react';
import { TranslationBox } from '../components/TranslationBox';
import { TranslateButton } from '../components/TranslateButton';

export function Translator({
  sourceLang,
  setSourceLang,
  targetLang,
  setTargetLang,
  inputText,
  setInputText,
  outputText,
  isLoading,
  onTranslate,
  onSwapLanguages,
  onClear
}) {
  const hasContent = inputText.trim().length > 0 || outputText.trim().length > 0;

  return (
    <div className="flex-1 flex flex-col max-w-6xl mx-auto w-full px-8 py-6 justify-between gap-8">
      {/* Title & Empty State Heading */}
      <div className="text-center my-2">
        <h2 className="text-3xl font-semibold text-white tracking-tight flex items-center justify-center gap-2">
          <span>Translate anything</span>
          <Sparkles className="w-5 h-5 text-purple-400" />
        </h2>
        <p className="text-sm text-gray-400 mt-2">
          Enter text on the left and get an instant, context-aware translation.
        </p>
      </div>

      {/* Main Panels Grid */}
      <div className="flex flex-col md:flex-row items-center gap-4 relative">
        {/* Source Panel */}
        <TranslationBox
          label="From"
          language={sourceLang}
          onLanguageChange={setSourceLang}
          text={inputText}
          onTextChange={setInputText}
          placeholder="Type text to translate..."
          onClear={onClear}
        />

        {/* Swap Languages Button */}
        <button
          onClick={onSwapLanguages}
          className="p-3.5 rounded-full bg-[#15151A] hover:bg-[#1D1D24] text-gray-300 hover:text-white border border-white/10 hover:border-purple-500/40 transition-all duration-200 shadow-xl z-10 hover:rotate-180"
          title="Swap source and target languages"
        >
          <ArrowLeftRight className="w-5 h-5" />
        </button>

        {/* Target Panel */}
        <TranslationBox
          label="To"
          language={targetLang}
          onLanguageChange={setTargetLang}
          text={outputText}
          isReadOnly={true}
          placeholder="Translation will appear here..."
          onRegenerate={onTranslate}
        />
      </div>

      {/* Action Button Container */}
      <div className="flex justify-center my-4">
        <TranslateButton
          onClick={onTranslate}
          isLoading={isLoading}
          disabled={!inputText.trim()}
        />
      </div>
    </div>
  );
}
