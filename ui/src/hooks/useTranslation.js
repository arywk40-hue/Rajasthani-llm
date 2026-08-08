import { useState, useCallback } from 'react';
import { translateText } from '../services/translation';

export function useTranslation() {
  const [sourceLang, setSourceLang] = useState('hi');
  const [targetLang, setTargetLang] = useState('marwari');
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([
    { id: 1, sourceLang: 'hi', targetLang: 'en', input: 'नमस्ते, आप कैसे हैं?', output: 'Hello, how are you?' },
    { id: 2, sourceLang: 'en', targetLang: 'marwari', input: 'Welcome to Rajasthan', output: 'राजस्थान में आपरो स्वागत है' },
    { id: 3, sourceLang: 'hi', targetLang: 'mewari', input: 'सब ठीक है', output: 'सब चोखो है' },
  ]);

  const handleTranslate = useCallback(async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    try {
      const result = await translateText(sourceLang, targetLang, inputText);
      setOutputText(result.translation);

      // Add to history
      const newHistoryItem = {
        id: Date.now(),
        sourceLang,
        targetLang,
        input: inputText,
        output: result.translation,
      };
      setHistory(prev => [newHistoryItem, ...prev.slice(0, 19)]);
    } catch (err) {
      console.error('Translation Hook Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, sourceLang, targetLang]);

  const handleSwapLanguages = useCallback(() => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setInputText(outputText);
    setOutputText(inputText);
  }, [sourceLang, targetLang, inputText, outputText]);

  const handleClear = useCallback(() => {
    setInputText('');
    setOutputText('');
  }, []);

  const selectHistoryItem = useCallback((item) => {
    setSourceLang(item.sourceLang);
    setTargetLang(item.targetLang);
    setInputText(item.input);
    setOutputText(item.output);
  }, []);

  return {
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
  };
}
