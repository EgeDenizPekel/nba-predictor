import { useState } from 'react';
import GamesTab from './components/GamesTab';
import AnalysisTab from './components/AnalysisTab';

type Tab = 'games' | 'analysis';

export default function App() {
  const [tab, setTab] = useState<Tab>('games');
  const [noSpoiler, setNoSpoiler] = useState(false);

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

          {/* No Spoiler toggle — centered */}
          {tab === 'games' && (
            <div className="flex-1 flex justify-center">
              <button
                onClick={() => setNoSpoiler((v) => !v)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors ${
                  noSpoiler
                    ? 'bg-amber-50 border-amber-300 text-amber-700'
                    : 'bg-white border-gray-200 text-gray-500 hover:border-gray-400'
                }`}
              >
                {noSpoiler ? (
                  // Eye-slash icon
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88L6.59 6.59m7.532 7.532l3.29 3.29M3 3l18 18" />
                  </svg>
                ) : (
                  // Eye icon
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                )}
                {noSpoiler ? 'No Spoiler: On' : 'No Spoiler'}
              </button>
            </div>
          )}

          {/* Legend — right side, hidden in no-spoiler mode */}
          {tab === 'games' && !noSpoiler && (
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

          {/* Spacer when no-spoiler hides the legend so the toggle stays centered */}
          {tab === 'games' && noSpoiler && <div className="ml-auto" />}
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {tab === 'games' ? (
          <GamesTab noSpoiler={noSpoiler} />
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
