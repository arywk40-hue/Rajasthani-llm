import React from 'react';
import { Settings, Moon, Sun } from 'lucide-react';

export function Header() {
  return (
    <header className="flex items-center justify-between px-8 py-6 border-b border-white/5">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-tight">Translate</h1>
        <p className="text-sm text-gray-400 mt-0.5">Translate text between Indian languages and English</p>
      </div>

      <div className="flex items-center gap-3">
        <button className="p-2 rounded-xl text-gray-400 hover:text-white hover:bg-[#15151A] border border-transparent hover:border-white/10 transition-colors">
          <Moon className="w-4 h-4" />
        </button>
        <button className="p-2 rounded-xl text-gray-400 hover:text-white hover:bg-[#15151A] border border-transparent hover:border-white/10 transition-colors">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
