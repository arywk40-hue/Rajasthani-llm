import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Translator } from './pages/Translator';
import { useTranslation } from './hooks/useTranslation';

export function App() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const {
    sourceLang,
    setSourceLang,
    targetLang,
    setTargetLang,
    inputText,
    setInputText,
    outputText,
    isLoading,
    history,
    handleTranslate,
    handleSwapLanguages,
    handleClear,
    selectHistoryItem
  } = useTranslation();

  return (
    <div className="flex h-screen w-screen bg-[#0B0B0F] text-gray-100 overflow-hidden font-sans gemini-ambient-bg">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        history={history}
        onSelectHistory={selectHistoryItem}
        onNewTranslation={handleClear}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-y-auto">
        <Header />
        
        <main className="flex-1 flex flex-col">
          <Translator
            sourceLang={sourceLang}
            setSourceLang={setSourceLang}
            targetLang={targetLang}
            setTargetLang={setTargetLang}
            inputText={inputText}
            setInputText={setInputText}
            outputText={outputText}
            isLoading={isLoading}
            onTranslate={handleTranslate}
            onSwapLanguages={handleSwapLanguages}
            onClear={handleClear}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
