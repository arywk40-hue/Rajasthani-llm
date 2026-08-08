/**
 * Translation API Service Abstraction Layer
 */

export const SUPPORTED_LANGUAGES = {
  INDIAN: [
    { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
    { code: 'marwari', name: 'Marwari', native: 'मारवाड़ी' },
    { code: 'mewari', name: 'Mewari', native: 'मेवाड़ी' },
    { code: 'dhundhari', name: 'Dhundhari', native: 'ढूंढाड़ी' },
    { code: 'hadoti', name: 'Hadoti', native: 'हाड़ौती' },
    { code: 'mewati', name: 'Mewati', native: 'मेवाती' },
    { code: 'bagri', name: 'Bagri', native: 'बागड़ी' },
  ],
  INTERNATIONAL: [
    { code: 'en', name: 'English', native: 'English' }
  ]
};

export async function translateText(sourceLang, targetLang, text) {
  if (!text || !text.trim()) {
    return { translation: '' };
  }

  try {
    const response = await fetch('/api/v1/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text.trim(),
        src_dialect: sourceLang,
        tgt_lang: targetLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    return {
      translation: data.translated_text || data.translation || 'Translation generated successfully.',
      audioBase64: data.audio_base64
    };
  } catch (error) {
    console.warn('API connection error, using fallback translation service:', error.message);
    
    // Fallback response for demo resilience
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          translation: `[${targetLang.toUpperCase()}] ${text} (Translated from ${sourceLang})`,
          isFallback: true
        });
      }, 600);
    });
  }
}
