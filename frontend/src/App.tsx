import { useState } from 'react';
import GamesTab from './components/GamesTab';
import AnalysisTab from './components/AnalysisTab';

type Tab = 'games' | 'analysis';

export default function App() {
  const [tab, setTab] = useState<Tab>('games');

  return (
    <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shrink-0 z-10">
        <div className="px-6 py-3 flex items-center gap-6">
          <div>
            <h1 className="text-base font-bold text-gray-900">NBA Predictor</h1>
            <p className="text-xs text-gray-500">2003-04 through 2021-22 &middot; XGBoost</p>
          </div>
          <nav className="flex gap-1">
            {(['games', 'analysis'] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  tab === t
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {t === 'games' ? 'Games' : 'Analysis'}
              </button>
            ))}
          </nav>
          {tab === 'games' && (
            <div className="ml-auto flex items-center gap-3 text-xs text-gray-400">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-green-400 inline-block" />
                Correct call
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-red-400 inline-block" />
                Wrong call
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-gray-300 inline-block" />
                Low confidence (&lt;60%)
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {tab === 'games' ? (
          <GamesTab />
        ) : (
          <div className="h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto px-6 py-6">
              <AnalysisTab />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
