import React from 'react';
import { SUPPORTED_LANGUAGES } from '../services/translation';

export function LanguageSelector({ value, onChange }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-[#1D1D24] text-white font-medium text-sm px-4 py-2 rounded-xl border border-white/10 outline-none cursor-pointer hover:border-purple-500/50 transition-all appearance-none pr-8"
      >
        <optgroup label="Indian Languages">
          {SUPPORTED_LANGUAGES.INDIAN.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} ({lang.native})
            </option>
          ))}
        </optgroup>
        <optgroup label="International">
          {SUPPORTED_LANGUAGES.INTERNATIONAL.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name}
            </option>
          ))}
        </optgroup>
      </select>
      <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400 text-xs">
        ▼
      </div>
    </div>
  );
}
