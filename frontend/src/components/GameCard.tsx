import type { GameSummary } from '../types';

interface Props {
  game: GameSummary;
  isSelected: boolean;
  noSpoiler?: boolean;
  onClick: () => void;
}

export default function GameCard({ game, isSelected, noSpoiler = false, onClick }: Props) {
  const prob = game.home_win_prob;
  const homeWon = game.home_win === 1;
  const correct = (prob >= 0.5 && homeWon) || (prob < 0.5 && !homeWon);
  const confident = Math.abs(prob - 0.5) > 0.1;

  const borderColor = isSelected
    ? 'border-blue-500'
    : noSpoiler
    ? 'border-gray-200'
    : confident && correct
    ? 'border-green-400'
    : confident && !correct
    ? 'border-red-400'
    : 'border-gray-200';

  // Bar color scales with confidence: muted when close to 50%, saturated when high
  const barColor =
    Math.abs(prob - 0.5) <= 0.1
      ? 'bg-blue-300'
      : Math.abs(prob - 0.5) <= 0.25
      ? 'bg-blue-500'
      : 'bg-blue-700';

  // Outcome icon
  const outcomeIcon = confident ? (
    correct ? (
      <svg className="w-3.5 h-3.5 text-green-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    ) : (
      <svg className="w-3.5 h-3.5 text-red-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    )
  ) : (
    <span className="w-3.5 h-3.5 flex items-center justify-center text-gray-300 text-xs font-bold shrink-0">—</span>
  );

  return (
    <button
      onClick={onClick}
      className={`w-full text-left border-2 rounded-lg px-3 py-2 bg-white hover:bg-gray-50 transition-colors ${borderColor}`}
    >
      <div className="flex items-center gap-2">
        {/* Teams — both neutral weight in no-spoiler mode */}
        <div className="w-10 shrink-0">
          <div className={`text-xs font-bold leading-tight ${!noSpoiler && homeWon ? 'text-gray-800' : noSpoiler ? 'text-gray-700' : 'text-gray-400'}`}>
            {game.home_team_abbr}
          </div>
          <div className={`text-xs font-bold leading-tight ${!noSpoiler && !homeWon ? 'text-gray-800' : noSpoiler ? 'text-gray-700' : 'text-gray-400'}`}>
            {game.away_team_abbr}
          </div>
        </div>

        {/* Prob bar with midpoint marker */}
        <div className="flex-1 relative h-1.5 opacity-70">
          <div className="absolute inset-0 rounded-full bg-gray-200 overflow-hidden">
            <div className={`h-full transition-all ${barColor}`} style={{ width: `${prob * 100}%` }} />
          </div>
          {/* 50% midpoint marker */}
          <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white opacity-70 z-10" />
        </div>

        {/* Home win % */}
        <div className="text-xs text-gray-500 w-7 text-right shrink-0 tabular-nums">
          {(prob * 100).toFixed(0)}%
        </div>

        {/* Outcome icon — hidden in no-spoiler mode */}
        {!noSpoiler && outcomeIcon}
      </div>

      {game.is_bubble_game === 1 && (
        <p className="text-xs text-orange-500 mt-0.5">Bubble</p>
      )}
      {game.is_no_fans_season === 1 && (
        <p className="text-xs text-red-500 mt-0.5">No fans</p>
      )}
    </button>
  );
}
