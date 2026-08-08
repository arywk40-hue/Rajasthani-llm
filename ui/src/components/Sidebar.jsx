import React from 'react';
import { Plus, History as HistoryIcon, Settings, Sparkles, ChevronLeft, ChevronRight, Languages } from 'lucide-react';
import { SUPPORTED_LANGUAGES } from '../services/translation';

const ALL_LANGS = [...SUPPORTED_LANGUAGES.INDIAN, ...SUPPORTED_LANGUAGES.INTERNATIONAL];

function getLangName(code) {
  const match = ALL_LANGS.find(l => l.code === code);
  return match ? match.name : code.toUpperCase();
}

export function Sidebar({ isCollapsed, setIsCollapsed, history, onSelectHistory, onNewTranslation }) {
  return (
    <aside className={`bg-[#15151A] border-r border-white/10 flex flex-col justify-between transition-all duration-300 ${isCollapsed ? 'w-16' : 'w-64'} h-screen z-20`}>
      {/* Top Brand & New Action */}
      <div className="p-4 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-purple-600/30 to-blue-500/30 border border-purple-500/30 text-purple-400">
              <Sparkles className="w-5 h-5" />
            </div>
            {!isCollapsed && (
              <span className="font-semibold text-white tracking-wide text-base">
                Translate
              </span>
            )}
          </div>
          
          <button 
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-[#1D1D24] transition-colors"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* New Translation Button */}
        <button
          onClick={onNewTranslation}
          className={`flex items-center gap-3 bg-[#1D1D24] hover:bg-[#252530] text-white border border-white/10 rounded-xl p-3 transition-all ${isCollapsed ? 'justify-center' : ''}`}
        >
          <Plus className="w-4 h-4 text-purple-400" />
          {!isCollapsed && <span className="text-sm font-medium">New Translation</span>}
        </button>

        {/* History List */}
        {!isCollapsed && (
          <div className="flex flex-col gap-2 mt-2 overflow-hidden">
            <div className="flex items-center gap-2 px-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
              <HistoryIcon className="w-3.5 h-3.5" />
              <span>History</span>
            </div>

            <div className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100vh-280px)] pr-1">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onSelectHistory(item)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-gray-300 hover:text-white hover:bg-[#1D1D24] text-left transition-colors truncate border border-transparent hover:border-white/5"
                >
                  <Languages className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                  <span className="truncate">
                    {getLangName(item.sourceLang)} → {getLangName(item.targetLang)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Bottom Settings */}
      <div className="p-4 border-t border-white/10">
        <button className={`flex items-center gap-3 text-gray-400 hover:text-white p-2 rounded-lg hover:bg-[#1D1D24] w-full transition-colors ${isCollapsed ? 'justify-center' : ''}`}>
          <Settings className="w-4 h-4" />
          {!isCollapsed && <span className="text-xs font-medium">Settings</span>}
        </button>
      </div>
    </aside>
  );
}
