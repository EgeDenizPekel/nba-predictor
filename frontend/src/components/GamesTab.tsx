import { useEffect, useState, type ReactNode } from 'react';
import type { GameSummary } from '../types';
import { api } from '../api';
import { fmtTeamName } from '../utils';
import GameCard from './GameCard';
import GameDetail from './GameDetail';
import CalendarPicker from './CalendarPicker';

const DEFAULT_DATE = '2021-11-15';

type Filter = 'all' | 'correct' | 'wrong' | 'uncertain' | 'high-conf-wrong';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'correct', label: 'Correct' },
  { key: 'wrong', label: 'Wrong' },
  { key: 'uncertain', label: 'Uncertain' },
  { key: 'high-conf-wrong', label: 'Big Misses' },
];

function classifyGame(game: GameSummary): 'correct' | 'wrong' | 'uncertain' {
  const confident = Math.abs(game.home_win_prob - 0.5) > 0.1;
  if (!confident) return 'uncertain';
  const correct =
    (game.home_win_prob >= 0.5 && game.home_win === 1) ||
    (game.home_win_prob < 0.5 && game.home_win === 0);
  return correct ? 'correct' : 'wrong';
}

interface Props {
  noSpoiler?: boolean;
}

export default function GamesTab({ noSpoiler = false }: Props) {
  const [date, setDate] = useState(DEFAULT_DATE);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  useEffect(() => {
    if (!date) return;
    setLoading(true);
    setError(null);
    setSelectedId(null);
    api
      .games(date)
      .then(setGames)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [date]);

  // When no-spoiler turns on, reset any outcome-revealing filter
  useEffect(() => {
    if (noSpoiler && filter !== 'all' && filter !== 'uncertain') {
      setFilter('all');
    }
  }, [noSpoiler]);

  const visibleGames =
    filter === 'all'
      ? games
      : filter === 'high-conf-wrong'
      ? games.filter(
          (g) =>
            classifyGame(g) === 'wrong' && Math.abs(g.home_win_prob - 0.5) > 0.2,
        )
      : games.filter((g) => classifyGame(g) === filter);

  const selectedGame = games.find((g) => g.game_id === selectedId) ?? null;

  // Evaluation badge for the detail header
  const evalBadge = selectedGame
    ? (() => {
        const cls = classifyGame(selectedGame);
        const homeAbbr = selectedGame.home_team_abbr;
        const awayAbbr = selectedGame.away_team_abbr;
        const winner = selectedGame.home_win ? homeAbbr : awayAbbr;
        const score =
          selectedGame.home_pts !== null
            ? `${selectedGame.home_pts.toFixed(0)}-${selectedGame.away_pts!.toFixed(0)}`
            : null;

        // Show the probability for the team the model actually backed
        const predictedHome = selectedGame.home_win_prob >= 0.5;
        const predicted = predictedHome ? homeAbbr : awayAbbr;
        const predictedProb = predictedHome
          ? (selectedGame.home_win_prob * 100).toFixed(1)
          : ((1 - selectedGame.home_win_prob) * 100).toFixed(1);

        let badge: ReactNode;
        let tooltipText: string;

        if (cls === 'correct') {
          tooltipText = score
            ? `Predicted ${predicted} to win at ${predictedProb}%. ${winner} won ${score}.`
            : `Predicted ${predicted} to win at ${predictedProb}%. ${winner} won.`;
          badge = (
            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-green-100 text-green-700">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Model correct
            </span>
          );
        } else if (cls === 'wrong') {
          tooltipText = score
            ? `Predicted ${predicted} to win at ${predictedProb}%. ${winner} won ${score}.`
            : `Predicted ${predicted} to win at ${predictedProb}%. ${winner} won.`;
          badge = (
            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-red-100 text-red-700">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
              Model wrong
            </span>
          );
        } else {
          tooltipText = `Predicted ${predictedProb}% — within 10% of 50%, below the confidence threshold.`;
          badge = (
            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-gray-100 text-gray-500">
              <span className="text-base leading-none">—</span>
              Low confidence
            </span>
          );
        }

        return (
          <div className="relative group inline-block">
            {badge}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block z-50 pointer-events-none">
              <div className="bg-gray-800 text-white text-xs rounded px-2.5 py-1.5 whitespace-nowrap shadow-lg">
                {tooltipText}
              </div>
              <div className="w-2 h-2 bg-gray-800 rotate-45 mx-auto -mt-1" />
            </div>
          </div>
        );
      })()
    : null;

  return (
    <div className="flex h-full">
      {/* Sidebar: calendar + filters + game list */}
      <aside className="w-72 shrink-0 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
        {/* Calendar - fixed at top */}
        <div className="p-3 border-b border-gray-100 shrink-0">
          <CalendarPicker value={date} onChange={setDate} />
          <div className="mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 text-center">
            {loading
              ? 'Loading…'
              : games.length > 0
              ? `${games.length} game${games.length !== 1 ? 's' : ''} on ${date}`
              : date
              ? `No games on ${date}`
              : ''}
          </div>
        </div>

        {/* Filter pills — outcome-revealing filters hidden in no-spoiler mode */}
        {games.length > 0 && (
          <div className="flex gap-1 px-3 py-2 border-b border-gray-100 shrink-0">
            {FILTERS
              .filter(({ key }) =>
                noSpoiler ? key === 'all' || key === 'uncertain' : true,
              )
              .map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => {
                    setFilter(key);
                    setSelectedId(null);
                  }}
                  className={`flex-1 text-xs py-1 rounded-full border transition-colors ${
                    filter === key
                      ? key === 'wrong' || key === 'high-conf-wrong'
                        ? 'bg-red-600 text-white border-red-600'
                        : 'bg-gray-800 text-white border-gray-800'
                      : 'bg-white text-gray-500 border-gray-200 hover:border-gray-400'
                  }`}
                >
                  {label}
                </button>
              ))}
          </div>
        )}

        {/* Game list - scrollable */}
        <div className="flex-1 overflow-y-auto p-3">
          {loading && (
            <p className="text-gray-400 text-sm text-center py-8">Loading…</p>
          )}
          {error && (
            <p className="text-red-500 text-sm text-center py-4">{error}</p>
          )}
          {!loading && games.length === 0 && date && !error && (
            <p className="text-gray-400 text-sm text-center py-8">
              No games on this date. Try another.
            </p>
          )}
          {!loading && games.length > 0 && visibleGames.length === 0 && (
            <p className="text-gray-400 text-sm text-center py-8">
              No {filter} predictions on this date.
            </p>
          )}
          <div className="flex flex-col gap-1.5">
            {visibleGames.map((game) => (
              <GameCard
                key={game.game_id}
                game={game}
                isSelected={game.game_id === selectedId}
                noSpoiler={noSpoiler}
                onClick={() =>
                  setSelectedId(game.game_id === selectedId ? null : game.game_id)
                }
              />
            ))}
          </div>
        </div>
      </aside>

      {/* Main: game detail */}
      <div className="flex-1 overflow-y-auto">
        {selectedGame ? (
          <div className="p-6">
            {/* Prediction experiment header */}
            <div className="mb-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                {fmtTeamName(selectedGame.home_team_abbr)} vs{' '}
                {fmtTeamName(selectedGame.away_team_abbr)}
              </h2>

              {/* Prediction */}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide w-24 shrink-0">Prediction</span>
                <span className="text-4xl font-bold text-blue-600 leading-none">
                  {(selectedGame.home_win_prob * 100).toFixed(1)}%
                </span>
                <span className="text-sm text-gray-400 self-end pb-0.5">P(Home Win)</span>
              </div>

              {/* Outcome — hidden in no-spoiler mode */}
              {!noSpoiler && (
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide w-24 shrink-0">Outcome</span>
                  {selectedGame.home_pts !== null ? (
                    <span className="text-sm text-gray-700 font-medium">
                      {selectedGame.home_pts.toFixed(0)}–{selectedGame.away_pts!.toFixed(0)}
                      <span className="text-gray-400 mx-1">·</span>
                      {selectedGame.home_win
                        ? `${selectedGame.home_team_abbr} won`
                        : `${selectedGame.away_team_abbr} won`}
                      <span className="text-gray-400 mx-1">·</span>
                      {selectedGame.season}
                    </span>
                  ) : (
                    <span className="text-sm text-gray-400">No result · {selectedGame.season}</span>
                  )}
                </div>
              )}

              {/* Evaluation — hidden in no-spoiler mode */}
              {!noSpoiler && (
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide w-24 shrink-0">Evaluation</span>
                  {evalBadge}
                </div>
              )}

              {/* No-spoiler season label replaces outcome row */}
              {noSpoiler && (
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide w-24 shrink-0">Season</span>
                  <span className="text-sm text-gray-500">{selectedGame.season}</span>
                </div>
              )}
            </div>

            <GameDetail
              gameId={selectedGame.game_id}
              homeAbbr={selectedGame.home_team_abbr}
              awayAbbr={selectedGame.away_team_abbr}
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <svg className="w-10 h-10 mb-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p className="text-sm">Click a game to see the feature breakdown</p>
          </div>
        )}
      </div>
    </div>
  );
}
